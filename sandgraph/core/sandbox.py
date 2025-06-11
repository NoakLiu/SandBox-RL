"""
沙盒抽象基类定义

基于 Game24bootcamp 的设计模式，每个沙盒都提供三个核心接口：
1. case_generator() - 生成任务实例
2. prompt_func() - 转换为自然语言提示
3. verify_score() - 验证结果并评分
"""

from typing import Any, Dict, Protocol
from abc import ABC, abstractmethod


class SandboxProtocol(Protocol):
    """沙盒协议定义，所有沙盒都应遵循此接口"""
    
    def case_generator(self) -> Dict[str, Any]:
        """生成一个任务实例
        
        Returns:
            Dict[str, Any]: 包含任务参数的字典
        """
        ...
    
    def prompt_func(self, case: Dict[str, Any]) -> str:
        """将任务实例转换为自然语言提示
        
        Args:
            case: case_generator()返回的任务实例
            
        Returns:
            str: 格式化的提示文本，可直接发送给LLM
        """
        ...
    
    def verify_score(self, response: str, case: Dict[str, Any], format_score: float = 0.0) -> float:
        """验证LLM响应并评分
        
        Args:
            response: LLM的响应文本
            case: 对应的任务实例
            format_score: 格式正确但答案错误时的得分
            
        Returns:
            float: 评分结果，通常0.0表示错误，1.0表示正确
        """
        ...


class Sandbox(ABC):
    """沙盒抽象基类
    
    所有具体沙盒实现都应继承此类并实现三个核心方法
    """
    
    def __init__(self, sandbox_id: str, description: str = ""):
        """初始化沙盒
        
        Args:
            sandbox_id: 沙盒唯一标识符
            description: 沙盒描述信息
        """
        self.sandbox_id = sandbox_id
        self.description = description
    
    @abstractmethod
    def case_generator(self) -> Dict[str, Any]:
        """生成任务实例 - 抽象方法，子类必须实现"""
        pass
    
    @abstractmethod
    def prompt_func(self, case: Dict[str, Any]) -> str:
        """构造提示文本 - 抽象方法，子类必须实现"""
        pass
    
    @abstractmethod
    def verify_score(self, response: str, case: Dict[str, Any], format_score: float = 0.0) -> float:
        """验证评分 - 抽象方法，子类必须实现"""
        pass
    
    def run_full_cycle(self, llm_response_generator=None) -> Dict[str, Any]:
        """执行完整的沙盒循环：生成任务 -> 构造提示 -> 评分
        
        Args:
            llm_response_generator: 可选的LLM响应生成器函数
            
        Returns:
            Dict包含case, prompt, response, score等信息
        """
        # 生成任务
        case = self.case_generator()
        
        # 构造提示
        prompt = self.prompt_func(case)
        
        # 如果提供了LLM响应生成器，获取响应并评分
        result = {
            "sandbox_id": self.sandbox_id,
            "case": case,
            "prompt": prompt
        }
        
        if llm_response_generator:
            response = llm_response_generator(prompt)
            score = self.verify_score(response, case)
            result.update({
                "response": response,
                "score": score
            })
        
        return result
    
    def __repr__(self) -> str:
        return f"Sandbox(id='{self.sandbox_id}', description='{self.description}')" 