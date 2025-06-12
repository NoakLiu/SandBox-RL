"""
大语言模型接口

提供统一的LLM接口，支持参数共享和强化学习优化
"""

from typing import Any, Dict, List, Optional, Callable
from abc import ABC, abstractmethod
import logging
import time
import threading
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """LLM响应结果"""
    text: str
    confidence: float = 0.0
    reasoning: str = ""
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseLLM(ABC):
    """基础LLM抽象类"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """生成响应"""
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """获取模型参数"""
        pass
    
    @abstractmethod
    def update_parameters(self, gradients: Dict[str, Any], learning_rate: float = 1e-4) -> None:
        """更新模型参数"""
        pass


class MockLLM(BaseLLM):
    """模拟LLM实现（用于演示）"""
    
    def __init__(self, model_name: str = "mock_llm"):
        super().__init__(model_name)
        self.parameters = {
            "embedding_weights": [0.1] * 1000,  # 模拟嵌入层权重
            "attention_weights": [0.2] * 500,   # 模拟注意力权重
            "output_weights": [0.3] * 200       # 模拟输出层权重
        }
        self.generation_count = 0
        self.update_count = 0
        self.lock = threading.Lock()
        
        # 模拟不同类型的推理能力
        self.reasoning_templates = {
            "mathematical": "数学推理：分析问题 → 建立方程 → 求解 → 验证",
            "logical": "逻辑推理：前提分析 → 规则应用 → 结论推导 → 一致性检查",
            "strategic": "策略推理：目标分析 → 选项评估 → 风险评估 → 最优选择",
            "creative": "创造性推理：问题理解 → 发散思考 → 方案生成 → 可行性评估"
        }
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """生成响应"""
        with self.lock:
            self.generation_count += 1
            
            # 模拟推理过程
            reasoning_type = kwargs.get("reasoning_type", "logical")
            temperature = kwargs.get("temperature", 0.7)
            max_tokens = kwargs.get("max_tokens", 100)
            
            # 基于prompt内容生成响应
            if "数学" in prompt or "计算" in prompt or "24点" in prompt:
                reasoning = self.reasoning_templates["mathematical"]
                response_text = f"基于数学推理，我分析了问题并得出结论。温度参数: {temperature}"
                confidence = 0.8 + (self.update_count * 0.01)  # 随训练提升
            elif "策略" in prompt or "规划" in prompt or "选择" in prompt:
                reasoning = self.reasoning_templates["strategic"]
                response_text = f"通过策略分析，我制定了最优方案。参数更新次数: {self.update_count}"
                confidence = 0.7 + (self.update_count * 0.015)
            elif "创新" in prompt or "创造" in prompt:
                reasoning = self.reasoning_templates["creative"]
                response_text = f"运用创造性思维，我提出了新的解决方案。"
                confidence = 0.6 + (self.update_count * 0.02)
            else:
                reasoning = self.reasoning_templates["logical"]
                response_text = f"通过逻辑推理，我得出了合理的结论。生成次数: {self.generation_count}"
                confidence = 0.75 + (self.update_count * 0.012)
            
            # 限制置信度范围
            confidence = min(0.95, confidence)
            
            return LLMResponse(
                text=response_text,
                confidence=confidence,
                reasoning=reasoning,
                metadata={
                    "generation_count": self.generation_count,
                    "update_count": self.update_count,
                    "temperature": temperature,
                    "reasoning_type": reasoning_type,
                    "prompt_length": len(prompt)
                }
            )
    
    def get_parameters(self) -> Dict[str, Any]:
        """获取模型参数"""
        with self.lock:
            return {
                "parameters": self.parameters.copy(),
                "generation_count": self.generation_count,
                "update_count": self.update_count,
                "model_name": self.model_name
            }
    
    def update_parameters(self, gradients: Dict[str, Any], learning_rate: float = 1e-4) -> None:
        """更新模型参数"""
        with self.lock:
            # 模拟参数更新
            for param_name, gradient in gradients.items():
                if param_name in self.parameters:
                    # 简化的梯度下降更新
                    if isinstance(gradient, (list, tuple)):
                        for i in range(min(len(self.parameters[param_name]), len(gradient))):
                            self.parameters[param_name][i] -= learning_rate * gradient[i]
                    else:
                        # 标量梯度，应用到所有参数
                        for i in range(len(self.parameters[param_name])):
                            self.parameters[param_name][i] -= learning_rate * gradient
            
            self.update_count += 1
            logger.info(f"LLM参数更新完成，更新次数: {self.update_count}")


class SharedLLMManager:
    """共享LLM管理器 - 全局只有一个LLM实例"""
    
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.lock = threading.Lock()
        
        # 节点注册管理
        self.registered_nodes: Dict[str, Dict[str, Any]] = {}
        self.node_usage_stats: Dict[str, Dict[str, Any]] = {}
        
        # 全局统计
        self.total_generations = 0
        self.total_updates = 0
        
    def register_node(self, node_id: str, node_config: Optional[Dict[str, Any]] = None) -> None:
        """注册使用LLM的节点"""
        if node_config is None:
            node_config = {}
            
        with self.lock:
            self.registered_nodes[node_id] = {
                "config": node_config,
                "registered_time": time.time()
            }
            self.node_usage_stats[node_id] = {
                "generation_count": 0,
                "last_used": None,
                "total_tokens": 0
            }
            logger.info(f"注册LLM节点: {node_id}")
    
    def generate_for_node(self, node_id: str, prompt: str, **kwargs) -> LLMResponse:
        """为特定节点生成响应"""
        with self.lock:
            if node_id not in self.registered_nodes:
                raise ValueError(f"节点 {node_id} 未注册")
            
            # 合并节点配置和调用参数
            node_config = self.registered_nodes[node_id]["config"]
            merged_kwargs = {**node_config, **kwargs}
            
            # 调用共享LLM
            response = self.llm.generate(prompt, **merged_kwargs)
            
            # 更新统计
            self.node_usage_stats[node_id]["generation_count"] += 1
            self.node_usage_stats[node_id]["last_used"] = time.time()
            self.node_usage_stats[node_id]["total_tokens"] += len(response.text.split())
            self.total_generations += 1
            
            # 在响应中添加节点信息
            if response.metadata:
                response.metadata["node_id"] = node_id
                response.metadata["global_generation_count"] = self.total_generations
            
            return response
    
    def update_shared_parameters(self, gradients: Dict[str, Any], learning_rate: float = 1e-4) -> Dict[str, Any]:
        """更新共享LLM参数"""
        with self.lock:
            self.llm.update_parameters(gradients, learning_rate)
            self.total_updates += 1
            
            return {
                "status": "updated",
                "update_count": self.total_updates,
                "affected_nodes": list(self.registered_nodes.keys()),
                "learning_rate": learning_rate
            }
    
    def get_global_stats(self) -> Dict[str, Any]:
        """获取全局统计信息"""
        with self.lock:
            llm_params = self.llm.get_parameters()
            
            return {
                "llm_model": self.llm.model_name,
                "total_generations": self.total_generations,
                "total_updates": self.total_updates,
                "registered_nodes_count": len(self.registered_nodes),
                "node_usage_stats": self.node_usage_stats.copy(),
                "llm_internal_stats": {
                    "generation_count": llm_params.get("generation_count", 0),
                    "update_count": llm_params.get("update_count", 0)
                }
            }
    
    def get_node_stats(self, node_id: str) -> Dict[str, Any]:
        """获取特定节点的统计信息"""
        with self.lock:
            if node_id not in self.registered_nodes:
                raise ValueError(f"节点 {node_id} 未注册")
            
            return {
                "node_id": node_id,
                "config": self.registered_nodes[node_id]["config"],
                "usage_stats": self.node_usage_stats[node_id].copy(),
                "shared_llm_model": self.llm.model_name
            }


# 便利函数
def create_shared_llm_manager(model_name: str = "global_shared_llm") -> SharedLLMManager:
    """创建共享LLM管理器"""
    llm = MockLLM(model_name)
    return SharedLLMManager(llm) 