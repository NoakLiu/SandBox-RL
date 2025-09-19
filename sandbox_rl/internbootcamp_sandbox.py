"""
InternBootcamp 沙盒实现

基于上海AI实验室的 InternBootcamp 项目，提供标准化的推理训练沙盒环境。
支持多种类型的推理任务，包括逻辑推理、算法问题、编程挑战等。

项目地址：https://github.com/InternLM/InternBootcamp
"""

import random
import re
import json
import ast
from typing import Any, Dict, List, Optional, Union
from abc import ABC, abstractmethod
from .core.sandbox import Sandbox

try:
    # 尝试导入internbootcamp（如果已安装）
    from internbootcamp import BaseBootcamp
    INTERNBOOTCAMP_AVAILABLE = True
except ImportError:
    # 如果没有安装，提供基础实现
    INTERNBOOTCAMP_AVAILABLE = False
    
    class BaseBootcamp:
        def __init__(self, **kwargs):
            pass
        
        def case_generator(self):
            return {}
        
        def prompt_func(self, case):
            return ""
        
        def verify_score(self, response, case, format_score=0.0):
            return 0.0


class InternBootcampBaseSandbox(Sandbox):
    """InternBootcamp基础沙盒抽象类"""
    
    def __init__(self, sandbox_id: str, description: str = "", **bootcamp_kwargs):
        """初始化InternBootcamp沙盒
        
        Args:
            sandbox_id: 沙盒唯一标识符
            description: 沙盒描述
            **bootcamp_kwargs: 传递给具体bootcamp的参数
        """
        super().__init__(sandbox_id, description)
        self.bootcamp_kwargs = bootcamp_kwargs
        self._bootcamp = None
        
        if INTERNBOOTCAMP_AVAILABLE:
            self._initialize_bootcamp()
        else:
            print(f"警告：InternBootcamp未安装，使用模拟实现。请运行: pip install git+https://github.com/InternLM/InternBootcamp.git")
    
    def _initialize_bootcamp(self):
        """初始化具体的bootcamp实例（子类实现）"""
        pass
    
    def case_generator(self) -> Dict[str, Any]:
        """生成任务实例"""
        if self._bootcamp and INTERNBOOTCAMP_AVAILABLE:
            return self._bootcamp.case_generator()
        else:
            return self._get_mock_case()
    
    def prompt_func(self, case: Dict[str, Any]) -> str:
        """构造提示文本"""
        if self._bootcamp and INTERNBOOTCAMP_AVAILABLE:
            return self._bootcamp.prompt_func(case)
        else:
            return self._get_mock_prompt(case)
    
    def verify_score(self, response: str, case: Dict[str, Any], format_score: float = 0.0) -> float:
        """验证评分"""
        if self._bootcamp and INTERNBOOTCAMP_AVAILABLE:
            return self._bootcamp.verify_score(response, case, format_score)
        else:
            return self._get_mock_score(response, case, format_score)
    
    @abstractmethod
    def _get_mock_case(self) -> Dict[str, Any]:
        """获取模拟任务（在没有internbootcamp时使用）"""
        pass
    
    @abstractmethod
    def _get_mock_prompt(self, case: Dict[str, Any]) -> str:
        """获取模拟提示（在没有internbootcamp时使用）"""
        pass
    
    @abstractmethod
    def _get_mock_score(self, response: str, case: Dict[str, Any], format_score: float) -> float:
        """获取模拟评分（在没有internbootcamp时使用）"""
        pass


class Game24BootcampSandbox(InternBootcampBaseSandbox):
    """Game24 Bootcamp沙盒
    
    基于InternBootcamp的Game24实现，提供算术谜题求解任务
    """
    
    def __init__(self, num_numbers: int = 4, range_max: int = 100, target_max: int = 100, seed: int = 42):
        super().__init__(
            "internbootcamp_game24", 
            "基于InternBootcamp的Game24算术谜题沙盒",
            num_numbers=num_numbers,
            range_max=range_max, 
            target_max=target_max,
            seed=seed
        )
        self.num_numbers = num_numbers
        self.range_max = range_max
        self.target_max = target_max
        self.random = random.Random(seed)
    
    def _initialize_bootcamp(self):
        """初始化Game24 bootcamp"""
        if INTERNBOOTCAMP_AVAILABLE:
            try:
                from internbootcamp import Game24Bootcamp
                self._bootcamp = Game24Bootcamp(**self.bootcamp_kwargs)
            except ImportError:
                print("警告：无法导入Game24Bootcamp，使用模拟实现")
                self._bootcamp = None
    
    def _get_mock_case(self) -> Dict[str, Any]:
        """模拟Game24任务生成"""
        nums = [self.random.randint(1, self.range_max) for _ in range(self.num_numbers)]
        target = 24  # Game24的标准目标是24
        return {"puzzle": nums, "target": target}
    
    def _get_mock_prompt(self, case: Dict[str, Any]) -> str:
        """模拟Game24提示生成"""
        nums = " ".join(map(str, case["puzzle"]))
        target = case["target"]
        return (
            f"请解决以下算术谜题：使用数字 {nums} 通过基本运算（+、-、×、÷）得到目标值 {target}。\n"
            f"请逐步思考并在最后用 \\boxed{{}} 格式给出答案。\n"
            f"最终答案应包含所有输入数字和运算符，可以使用括号改变运算顺序。\n"
            f"例如：\\boxed{{(6+6)+(6+6)}}"
        )
    
    def _get_mock_score(self, response: str, case: Dict[str, Any], format_score: float) -> float:
        """模拟Game24评分"""
        try:
            # 提取boxed内容
            boxed_matches = re.findall(r"\\boxed\{([^}]*)\}", response)
            if not boxed_matches:
                return format_score if "\\boxed{" in response else 0.0
            
            expr = boxed_matches[-1]
            expr = expr.replace("×", "*").replace("÷", "/")
            
            # 提取数字并检查
            nums_in_expr = list(map(int, re.findall(r"\d+", expr)))
            expected_nums = sorted(case["puzzle"])
            used_nums = sorted(nums_in_expr)
            
            if used_nums != expected_nums:
                return 0.0
            
            # 计算结果
            result = eval(expr)
            return 1.0 if abs(result - case["target"]) < 1e-6 else 0.0
            
        except Exception:
            return format_score if "\\boxed{" in response else 0.0


class ARCBootcampSandbox(InternBootcampBaseSandbox):
    """ARC-AGI Bootcamp沙盒
    
    基于Abstract Reasoning Corpus的视觉推理任务
    """
    
    def __init__(self, difficulty: str = "medium", seed: int = 42):
        super().__init__(
            "internbootcamp_arc", 
            "基于InternBootcamp的ARC-AGI视觉推理沙盒",
            difficulty=difficulty,
            seed=seed
        )
        self.difficulty = difficulty
        self.random = random.Random(seed)
    
    def _initialize_bootcamp(self):
        """初始化ARC bootcamp"""
        if INTERNBOOTCAMP_AVAILABLE:
            try:
                from internbootcamp import ArcBootcamp
                self._bootcamp = ArcBootcamp(**self.bootcamp_kwargs)
            except ImportError:
                print("警告：无法导入ArcBootcamp，使用模拟实现")
                self._bootcamp = None
    
    def _get_mock_case(self) -> Dict[str, Any]:
        """模拟ARC任务生成"""
        # 创建简单的3x3网格模式识别任务
        pattern = [
            [1, 0, 1],
            [0, 1, 0], 
            [1, 0, 1]
        ]
        return {
            "train_examples": [
                {"input": pattern, "output": [[0, 1, 0], [1, 0, 1], [0, 1, 0]]}
            ],
            "test_input": pattern,
            "difficulty": self.difficulty
        }
    
    def _get_mock_prompt(self, case: Dict[str, Any]) -> str:
        """模拟ARC提示生成"""
        return (
            "这是一个抽象推理任务（ARC-AGI）。\n"
            "给定输入输出示例，请找出规律并预测测试输入的输出。\n"
            f"训练示例：{json.dumps(case['train_examples'], indent=2)}\n"
            f"测试输入：{json.dumps(case['test_input'], indent=2)}\n"
            "请分析规律并给出测试输入对应的输出，格式为二维数组。"
        )
    
    def _get_mock_score(self, response: str, case: Dict[str, Any], format_score: float) -> float:
        """模拟ARC评分"""
        try:
            # 尝试从响应中提取数组
            lines = response.split('\n')
            for line in lines:
                if '[' in line and ']' in line:
                    try:
                        predicted = ast.literal_eval(line.strip())
                        if isinstance(predicted, list) and len(predicted) > 0:
                            return 0.8  # 基础分数，实际需要比较准确性
                    except:
                        continue
            return format_score
        except Exception:
            return 0.0


class KORBootcampSandbox(InternBootcampBaseSandbox):
    """KOR-Bench Bootcamp沙盒
    
    支持逻辑推理、操作推理、密码推理、谜题推理等多种推理类型
    """
    
    def __init__(self, reasoning_type: str = "logic", difficulty: str = "medium", seed: int = 42):
        """初始化KOR沙盒
        
        Args:
            reasoning_type: 推理类型 (logic, operation, cipher, puzzle)
            difficulty: 难度等级
            seed: 随机种子
        """
        super().__init__(
            f"internbootcamp_kor_{reasoning_type}", 
            f"基于InternBootcamp的KOR {reasoning_type} 推理沙盒",
            reasoning_type=reasoning_type,
            difficulty=difficulty,
            seed=seed
        )
        self.reasoning_type = reasoning_type
        self.difficulty = difficulty
        self.random = random.Random(seed)
    
    def _initialize_bootcamp(self):
        """初始化KOR bootcamp"""
        if INTERNBOOTCAMP_AVAILABLE:
            try:
                from internbootcamp import KORBootcamp
                self._bootcamp = KORBootcamp(**self.bootcamp_kwargs)
            except ImportError:
                print("警告：无法导入KORBootcamp，使用模拟实现")
                self._bootcamp = None
    
    def _get_mock_case(self) -> Dict[str, Any]:
        """模拟KOR任务生成"""
        if self.reasoning_type == "logic":
            return {
                "premise": "所有鸟类都会飞。企鹅是鸟类。",
                "question": "企鹅会飞吗？",
                "type": "logic_reasoning"
            }
        elif self.reasoning_type == "operation":
            return {
                "sequence": [2, 4, 6, 8],
                "question": "下一个数字是什么？",
                "type": "operation_reasoning"
            }
        elif self.reasoning_type == "cipher":
            return {
                "cipher_text": "URYYB",
                "hint": "每个字母向后移动13位",
                "question": "解密这个密码",
                "type": "cipher_reasoning"
            }
        else:  # puzzle
            return {
                "puzzle": "一个人在过桥时只能带一样东西。他需要带一只鸡、一袋谷子和一只狐狸过桥。",
                "constraint": "不能让狐狸和鸡单独在一起，也不能让鸡和谷子单独在一起",
                "question": "如何安排过桥顺序？",
                "type": "puzzle_reasoning"
            }
    
    def _get_mock_prompt(self, case: Dict[str, Any]) -> str:
        """模拟KOR提示生成"""
        reasoning_type = case.get("type", self.reasoning_type)
        
        if "logic" in reasoning_type:
            return (
                f"逻辑推理题：\n"
                f"前提：{case['premise']}\n"
                f"问题：{case['question']}\n"
                f"请进行逻辑分析并给出答案。"
            )
        elif "operation" in reasoning_type:
            return (
                f"操作推理题：\n"
                f"序列：{case['sequence']}\n"
                f"问题：{case['question']}\n"
                f"请找出规律并给出答案。"
            )
        elif "cipher" in reasoning_type:
            return (
                f"密码推理题：\n"
                f"密文：{case['cipher_text']}\n"
                f"提示：{case['hint']}\n"
                f"问题：{case['question']}\n"
                f"请解密并给出明文。"
            )
        else:  # puzzle
            return (
                f"谜题推理：\n"
                f"谜题：{case['puzzle']}\n"
                f"约束：{case['constraint']}\n"
                f"问题：{case['question']}\n"
                f"请分析并给出解决方案。"
            )
    
    def _get_mock_score(self, response: str, case: Dict[str, Any], format_score: float) -> float:
        """模拟KOR评分"""
        reasoning_type = case.get("type", self.reasoning_type)
        response_lower = response.lower()
        
        if "logic" in reasoning_type:
            # 检查是否包含逻辑分析关键词
            if any(word in response_lower for word in ["矛盾", "逻辑", "推理", "因此", "所以"]):
                return 0.8
        elif "operation" in reasoning_type:
            # 检查是否包含数字
            if re.search(r'\d+', response):
                return 0.7
        elif "cipher" in reasoning_type:
            # 检查是否包含解密结果
            if "hello" in response_lower or "你好" in response:
                return 1.0
        else:  # puzzle
            # 检查是否包含步骤分析
            if any(word in response_lower for word in ["第一步", "然后", "最后", "步骤"]):
                return 0.8
        
        return format_score


class AlgorithmBootcampSandbox(InternBootcampBaseSandbox):
    """算法问题 Bootcamp沙盒
    
    基于CodeContests等编程竞赛题目的算法推理任务
    """
    
    def __init__(self, difficulty: str = "medium", problem_type: str = "general", seed: int = 42):
        super().__init__(
            "internbootcamp_algorithm", 
            "基于InternBootcamp的算法问题沙盒",
            difficulty=difficulty,
            problem_type=problem_type,
            seed=seed
        )
        self.difficulty = difficulty
        self.problem_type = problem_type
        self.random = random.Random(seed)
    
    def _initialize_bootcamp(self):
        """初始化算法 bootcamp"""
        if INTERNBOOTCAMP_AVAILABLE:
            try:
                from internbootcamp import AlgorithmBootcamp
                self._bootcamp = AlgorithmBootcamp(**self.bootcamp_kwargs)
            except ImportError:
                print("警告：无法导入AlgorithmBootcamp，使用模拟实现")
                self._bootcamp = None
    
    def _get_mock_case(self) -> Dict[str, Any]:
        """模拟算法任务生成"""
        problems = [
            {
                "title": "最大子数组和",
                "description": "给定一个整数数组，找到其中和最大的连续子数组",
                "input": "数组: [-2,1,-3,4,-1,2,1,-5,4]",
                "expected_output": "6",
                "algorithm_type": "dynamic_programming"
            },
            {
                "title": "二分查找",
                "description": "在有序数组中查找目标值",
                "input": "数组: [1,3,5,7,9], 目标: 5",
                "expected_output": "2",
                "algorithm_type": "binary_search"
            },
            {
                "title": "图的最短路径",
                "description": "计算从起点到终点的最短路径长度",
                "input": "图: [[0,1,4],[1,2,2],[0,2,5]], 起点: 0, 终点: 2",
                "expected_output": "3",
                "algorithm_type": "graph"
            }
        ]
        return self.random.choice(problems)
    
    def _get_mock_prompt(self, case: Dict[str, Any]) -> str:
        """模拟算法提示生成"""
        return (
            f"算法问题：{case['title']}\n\n"
            f"问题描述：{case['description']}\n"
            f"输入：{case['input']}\n"
            f"请分析问题，选择合适的算法，并给出解决方案。\n"
            f"最终答案请用 \\boxed{{}} 格式给出。"
        )
    
    def _get_mock_score(self, response: str, case: Dict[str, Any], format_score: float) -> float:
        """模拟算法评分"""
        try:
            # 检查是否包含算法分析
            algorithm_keywords = ["算法", "复杂度", "时间", "空间", "遍历", "递归", "动态规划", "贪心"]
            has_analysis = any(keyword in response for keyword in algorithm_keywords)
            
            # 检查是否有正确答案
            boxed_matches = re.findall(r"\\boxed\{([^}]*)\}", response)
            if boxed_matches:
                answer = boxed_matches[-1].strip()
                expected = case.get("expected_output", "")
                if answer == expected:
                    return 1.0
                elif has_analysis:
                    return 0.6
            
            return 0.4 if has_analysis else format_score
            
        except Exception:
            return format_score


class ProgrammingBootcampSandbox(InternBootcampBaseSandbox):
    """编程能力 Bootcamp沙盒
    
    基于BigCodeBench、KodCode等编程基准测试
    """
    
    def __init__(self, language: str = "python", difficulty: str = "medium", seed: int = 42):
        super().__init__(
            "internbootcamp_programming", 
            "基于InternBootcamp的编程能力沙盒",
            language=language,
            difficulty=difficulty,
            seed=seed
        )
        self.language = language
        self.difficulty = difficulty
        self.random = random.Random(seed)
    
    def _initialize_bootcamp(self):
        """初始化编程 bootcamp"""
        if INTERNBOOTCAMP_AVAILABLE:
            try:
                from internbootcamp import ProgrammingBootcamp
                self._bootcamp = ProgrammingBootcamp(**self.bootcamp_kwargs)
            except ImportError:
                print("警告：无法导入ProgrammingBootcamp，使用模拟实现")
                self._bootcamp = None
    
    def _get_mock_case(self) -> Dict[str, Any]:
        """模拟编程任务生成"""
        problems = [
            {
                "function_name": "fibonacci",
                "description": "计算斐波那契数列的第n项",
                "signature": "def fibonacci(n: int) -> int:",
                "test_cases": [
                    {"input": [0], "expected": 0},
                    {"input": [1], "expected": 1},
                    {"input": [5], "expected": 5},
                    {"input": [10], "expected": 55}
                ]
            },
            {
                "function_name": "reverse_string",
                "description": "反转字符串",
                "signature": "def reverse_string(s: str) -> str:",
                "test_cases": [
                    {"input": ["hello"], "expected": "olleh"},
                    {"input": ["world"], "expected": "dlrow"},
                    {"input": [""], "expected": ""}
                ]
            }
        ]
        return self.random.choice(problems)
    
    def _get_mock_prompt(self, case: Dict[str, Any]) -> str:
        """模拟编程提示生成"""
        test_examples = "\n".join([
            f"  输入: {tc['input']}, 期望输出: {tc['expected']}"
            for tc in case['test_cases'][:2]  # 只显示前两个测试用例
        ])
        
        return (
            f"编程任务：{case['description']}\n\n"
            f"函数签名：{case['signature']}\n\n"
            f"测试样例：\n{test_examples}\n\n"
            f"请实现这个函数，确保能通过所有测试用例。"
        )
    
    def _get_mock_score(self, response: str, case: Dict[str, Any], format_score: float) -> float:
        """模拟编程评分"""
        try:
            # 检查是否包含函数定义
            if f"def {case['function_name']}" not in response:
                return format_score
            
            # 简单的代码质量检查
            code_quality_score = 0.5
            
            # 检查关键概念
            function_name = case['function_name']
            if "fibonacci" in function_name and any(keyword in response.lower() for keyword in ["递归", "循环", "递推"]):
                code_quality_score = 0.8
            elif "reverse" in function_name and any(keyword in response for keyword in ["[::-1]", "reverse", "倒序"]):
                code_quality_score = 0.9
            
            return code_quality_score
            
        except Exception:
            return format_score


# InternBootcamp沙盒注册表
INTERNBOOTCAMP_SANDBOX_REGISTRY = {
    "internbootcamp_game24": Game24BootcampSandbox,
    "internbootcamp_arc": ARCBootcampSandbox,
    "internbootcamp_kor_logic": lambda **kwargs: KORBootcampSandbox(reasoning_type="logic", **kwargs),
    "internbootcamp_kor_operation": lambda **kwargs: KORBootcampSandbox(reasoning_type="operation", **kwargs),
    "internbootcamp_kor_cipher": lambda **kwargs: KORBootcampSandbox(reasoning_type="cipher", **kwargs),
    "internbootcamp_kor_puzzle": lambda **kwargs: KORBootcampSandbox(reasoning_type="puzzle", **kwargs),
    "internbootcamp_algorithm": AlgorithmBootcampSandbox,
    "internbootcamp_programming": ProgrammingBootcampSandbox,
}


def create_internbootcamp_sandbox(sandbox_type: str, **kwargs) -> InternBootcampBaseSandbox:
    """创建InternBootcamp沙盒实例
    
    Args:
        sandbox_type: 沙盒类型
        **kwargs: 沙盒初始化参数
        
    Returns:
        InternBootcampBaseSandbox: 沙盒实例
    """
    if sandbox_type not in INTERNBOOTCAMP_SANDBOX_REGISTRY:
        raise ValueError(f"未知的InternBootcamp沙盒类型: {sandbox_type}. 支持的类型: {list(INTERNBOOTCAMP_SANDBOX_REGISTRY.keys())}")
    
    sandbox_class = INTERNBOOTCAMP_SANDBOX_REGISTRY[sandbox_type]
    return sandbox_class(**kwargs)


def list_internbootcamp_sandboxes() -> List[str]:
    """列出所有可用的InternBootcamp沙盒类型"""
    return list(INTERNBOOTCAMP_SANDBOX_REGISTRY.keys())


def get_internbootcamp_info() -> Dict[str, Any]:
    """获取InternBootcamp集成信息"""
    return {
        "available": INTERNBOOTCAMP_AVAILABLE,
        "supported_sandboxes": list_internbootcamp_sandboxes(),
        "message": "InternBootcamp SDK 可用" if INTERNBOOTCAMP_AVAILABLE else "InternBootcamp SDK 未安装，使用模拟实现",
        "installation_guide": "pip install git+https://github.com/InternLM/InternBootcamp.git" if not INTERNBOOTCAMP_AVAILABLE else None
    } 