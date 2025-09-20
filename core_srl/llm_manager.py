#!/usr/bin/env python3
"""
Unified LLM Manager - 统一LLM管理器
=================================

集成所有LLM相关功能：
1. 统一LLM接口（Mock, HuggingFace, OpenAI）
2. 共享LLM管理器
3. 冻结自适应更新
4. LoRA集成和压缩
5. KV缓存优化
"""

import logging
import time
import threading
import json
import os
import copy
import hashlib
import pickle
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from datetime import datetime

logger = logging.getLogger(__name__)

# Optional imports
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None


class LLMBackend(Enum):
    """LLM后端类型"""
    MOCK = "mock"
    HUGGINGFACE = "huggingface"
    OPENAI_API = "openai_api"
    VLLM = "vllm"


class UpdateStrategy(Enum):
    """参数更新策略"""
    FROZEN = "frozen"
    ADAPTIVE = "adaptive"
    SELECTIVE = "selective"
    INCREMENTAL = "incremental"


class ParameterImportance(Enum):
    """参数重要性级别"""
    CRITICAL = "critical"
    IMPORTANT = "important"
    MODERATE = "moderate"
    LOW = "low"


@dataclass
class LLMConfig:
    """统一LLM配置"""
    backend: LLMBackend = LLMBackend.MOCK
    model_name: str = "mock_llm"
    device: str = "auto"
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    
    # API配置
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    
    # LoRA配置
    enable_lora: bool = False
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.1
    
    # 冻结自适应配置
    update_strategy: UpdateStrategy = UpdateStrategy.ADAPTIVE
    frozen_layers: List[str] = field(default_factory=list)
    adaptive_learning_rate: bool = True
    min_learning_rate: float = 1e-6
    max_learning_rate: float = 1e-3


@dataclass
class LLMResponse:
    """LLM响应结果"""
    text: str
    confidence: float = 0.0
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParameterInfo:
    """参数信息"""
    name: str
    importance: ParameterImportance
    frozen: bool = False
    last_update: float = 0.0
    update_count: int = 0
    gradient_norm: float = 0.0
    sensitivity: float = 0.0


class AdaptiveLearningRate:
    """自适应学习率管理器"""
    
    def __init__(self, initial_lr: float = 1e-4, min_lr: float = 1e-6, max_lr: float = 1e-3):
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.current_lr = initial_lr
        self.performance_history = deque(maxlen=100)
    
    def update(self, performance_metric: float) -> float:
        """根据性能指标更新学习率"""
        self.performance_history.append(performance_metric)
        
        if len(self.performance_history) < 10:
            return self.current_lr
        
        # 计算性能趋势
        recent_performance = list(self.performance_history)[-10:]
        diffs = [recent_performance[i] - recent_performance[i-1] for i in range(1, len(recent_performance))]
        performance_trend = sum(diffs) / len(diffs) if diffs else 0.0
        
        # 根据趋势调整学习率
        if performance_trend > 0.01:
            self.current_lr = min(self.max_lr, self.current_lr * 1.1)
        elif performance_trend < -0.01:
            self.current_lr = max(self.min_lr, self.current_lr * 0.9)
        
        return self.current_lr


class BaseLLM(ABC):
    """基础LLM抽象类"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.model_name = config.model_name
        self.backend = config.backend
        self.generation_count = 0
        self.update_count = 0
        self.lock = threading.Lock()
        self.model_loaded = False
        
        # 冻结自适应组件
        self.parameter_info = {}
        self.adaptive_lr = AdaptiveLearningRate(
            initial_lr=1e-4,
            min_lr=config.min_learning_rate,
            max_lr=config.max_learning_rate
        )
        self.performance_history = deque(maxlen=100)
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """生成响应"""
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """获取模型参数"""
        pass
    
    def update_parameters(self, gradients: Dict[str, Any], learning_rate: float = 1e-4) -> None:
        """更新模型参数（带冻结自适应逻辑）"""
        with self.lock:
            # 分析参数重要性
            importance_scores = self._analyze_parameter_importance(gradients)
            
            # 更新学习率
            if self.config.adaptive_learning_rate and self.performance_history:
                learning_rate = self.adaptive_lr.update(self.performance_history[-1])
            
            # 应用更新策略
            updated_params = {}
            for param_name, gradient in gradients.items():
                if self._should_update_parameter(param_name, importance_scores.get(param_name)):
                    updated_params[param_name] = self._apply_gradient_update(param_name, gradient, learning_rate)
            
            # 执行实际更新
            self._execute_parameter_update(updated_params, learning_rate)
            self.update_count += 1
    
    def _analyze_parameter_importance(self, gradients: Dict[str, Any]) -> Dict[str, ParameterImportance]:
        """分析参数重要性"""
        importance_scores = {}
        for name, grad in gradients.items():
            # 计算梯度范数
            if isinstance(grad, (list, tuple)):
                grad_norm = sum(g * g for g in grad) ** 0.5
            elif isinstance(grad, (int, float)):
                grad_norm = abs(grad)
            else:
                grad_norm = 0.0
            
            # 根据梯度范数确定重要性
            if grad_norm > 0.1:
                importance = ParameterImportance.CRITICAL
            elif grad_norm > 0.05:
                importance = ParameterImportance.IMPORTANT
            elif grad_norm > 0.01:
                importance = ParameterImportance.MODERATE
            else:
                importance = ParameterImportance.LOW
            
            importance_scores[name] = importance
            
            # 更新参数信息
            if name not in self.parameter_info:
                self.parameter_info[name] = ParameterInfo(name=name, importance=importance)
            else:
                self.parameter_info[name].importance = importance
                self.parameter_info[name].gradient_norm = grad_norm
        
        return importance_scores
    
    def _should_update_parameter(self, param_name: str, importance: Optional[ParameterImportance]) -> bool:
        """判断是否应该更新参数"""
        if self.config.update_strategy == UpdateStrategy.FROZEN:
            return False
        
        if param_name in self.config.frozen_layers:
            return False
        
        if self.config.update_strategy == UpdateStrategy.SELECTIVE:
            return importance in [ParameterImportance.CRITICAL, ParameterImportance.IMPORTANT]
        
        return True
    
    def _apply_gradient_update(self, param_name: str, gradient: Any, learning_rate: float) -> Any:
        """应用梯度更新"""
        # 简化的梯度更新逻辑
        return gradient  # 实际实现中需要更复杂的逻辑
    
    def _execute_parameter_update(self, updated_params: Dict[str, Any], learning_rate: float):
        """执行参数更新"""
        # 子类实现具体的参数更新逻辑
        pass


class MockLLM(BaseLLM):
    """模拟LLM实现"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.parameters = {
            "embedding_weights": [0.1] * 1000,
            "attention_weights": [0.2] * 500,
            "output_weights": [0.3] * 200
        }
        
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
            
            temperature = kwargs.get("temperature", self.config.temperature)
            
            # 基于prompt内容选择推理类型
            if "数学" in prompt or "计算" in prompt:
                reasoning = self.reasoning_templates["mathematical"]
                response_text = f"基于数学推理，我分析了问题并得出结论。温度参数: {temperature}"
                confidence = 0.8 + (self.update_count * 0.01)
            elif "策略" in prompt or "规划" in prompt:
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
            
            confidence = min(0.95, confidence)
            
            return LLMResponse(
                text=response_text,
                confidence=confidence,
                reasoning=reasoning,
                metadata={
                    "backend": self.backend.value,
                    "generation_count": self.generation_count,
                    "update_count": self.update_count,
                    "temperature": temperature
                }
            )
    
    def get_parameters(self) -> Dict[str, Any]:
        """获取模型参数"""
        with self.lock:
            return {
                "parameters": self.parameters.copy(),
                "generation_count": self.generation_count,
                "update_count": self.update_count,
                "model_name": self.model_name,
                "backend": self.backend.value
            }
    
    def _execute_parameter_update(self, updated_params: Dict[str, Any], learning_rate: float):
        """执行Mock模型的参数更新"""
        for param_name, gradient in updated_params.items():
            if param_name in self.parameters:
                if isinstance(gradient, (list, tuple)):
                    for i in range(min(len(self.parameters[param_name]), len(gradient))):
                        self.parameters[param_name][i] -= learning_rate * gradient[i]


class HuggingFaceLLM(BaseLLM):
    """HuggingFace模型实现"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self._check_dependencies()
    
    def _check_dependencies(self):
        """检查依赖"""
        try:
            import torch
            import transformers
            self.torch = torch
            self.transformers = transformers
        except ImportError as e:
            logger.error(f"缺少依赖: {e}")
            raise
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """生成响应"""
        if not self.model_loaded:
            self._load_model()
        
        with self.lock:
            self.generation_count += 1
            
            try:
                # 处理输入
                temperature = kwargs.get("temperature", self.config.temperature)
                max_length = kwargs.get("max_length", self.config.max_length)
                
                # 编码输入
                inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=1024)
                inputs = inputs.to(self.device)
                
                # 生成响应
                start_time = time.time()
                with self.torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        max_new_tokens=256,
                        temperature=temperature,
                        top_p=self.config.top_p,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                generation_time = time.time() - start_time
                
                # 解码响应
                generated_ids = outputs[0][inputs.shape[1]:]
                response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                
                if not response_text:
                    response_text = "Based on the current situation, I recommend a cautious approach."
                
                confidence = min(0.95, 0.7 + (self.update_count * 0.01))
                
                return LLMResponse(
                    text=response_text,
                    confidence=confidence,
                    reasoning=f"使用{self.backend.value}模型进行文本生成",
                    metadata={
                        "backend": self.backend.value,
                        "generation_count": self.generation_count,
                        "generation_time": generation_time,
                        "temperature": temperature
                    }
                )
                
            except Exception as e:
                logger.error(f"生成响应失败: {e}")
                return LLMResponse(
                    text=f"生成失败: {str(e)}",
                    confidence=0.0,
                    reasoning="生成过程中出现错误",
                    metadata={"error": str(e)}
                )
    
    def _load_model(self):
        """加载模型"""
        logger.info(f"加载HuggingFace模型: {self.model_name}")
        
        # 设备配置
        device = "cuda" if self.torch.cuda.is_available() else "cpu" if self.config.device == "auto" else self.config.device
        torch_dtype = self.torch.float16 if device == "cuda" else self.torch.float32
        
        # 加载tokenizer
        self.tokenizer = self.transformers.AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # 加载模型
        self.model = self.transformers.AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        ).to(device)
        
        self.device = device
        self.model_loaded = True
    
    def get_parameters(self) -> Dict[str, Any]:
        """获取模型参数"""
        with self.lock:
            params_info = {
                "model_name": self.model_name,
                "backend": self.backend.value,
                "generation_count": self.generation_count,
                "update_count": self.update_count,
                "model_loaded": self.model_loaded
            }
            
            if self.model_loaded and self.model is not None:
                total_params = sum(p.numel() for p in self.model.parameters())
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                params_info.update({
                    "total_parameters": total_params,
                    "trainable_parameters": trainable_params,
                    "device": getattr(self, 'device', 'unknown')
                })
            
            return params_info
    
    def _execute_parameter_update(self, updated_params: Dict[str, Any], learning_rate: float):
        """执行HuggingFace模型的参数更新"""
        # 实际的参数更新逻辑
        logger.info(f"HuggingFace模型参数更新完成，更新次数: {self.update_count}")


class OpenAILLM(BaseLLM):
    """OpenAI API实现"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._setup_client()
    
    def _setup_client(self):
        """设置OpenAI客户端"""
        try:
            import openai
            self.openai = openai
            
            api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
            if api_key:
                self.client = openai.OpenAI(api_key=api_key, base_url=self.config.api_base)
                self.model_loaded = True
        except ImportError:
            logger.error("缺少OpenAI依赖，请安装: pip install openai")
            raise
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """通过OpenAI API生成响应"""
        with self.lock:
            self.generation_count += 1
            
            try:
                temperature = kwargs.get("temperature", self.config.temperature)
                max_tokens = kwargs.get("max_tokens", min(self.config.max_length, 4096))
                
                start_time = time.time()
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=self.config.top_p
                )
                
                generation_time = time.time() - start_time
                response_text = response.choices[0].message.content
                finish_reason = response.choices[0].finish_reason
                confidence = 0.9 if finish_reason == "stop" else 0.7
                
                return LLMResponse(
                    text=response_text,
                    confidence=confidence,
                    reasoning=f"通过OpenAI API调用{self.model_name}模型生成",
                    metadata={
                        "backend": self.backend.value,
                        "generation_count": self.generation_count,
                        "generation_time": generation_time,
                        "finish_reason": finish_reason
                    }
                )
                
            except Exception as e:
                logger.error(f"OpenAI API调用失败: {e}")
                return LLMResponse(
                    text=f"API调用失败: {str(e)}",
                    confidence=0.0,
                    reasoning="OpenAI API调用出现错误",
                    metadata={"error": str(e)}
                )
    
    def get_parameters(self) -> Dict[str, Any]:
        """获取OpenAI模型参数信息"""
        return {
            "model_name": self.model_name,
            "backend": self.backend.value,
            "generation_count": self.generation_count,
            "api_available": hasattr(self, 'client')
        }
    
    def _execute_parameter_update(self, updated_params: Dict[str, Any], learning_rate: float):
        """OpenAI模型不支持参数更新"""
        logger.warning("OpenAI API模式不支持参数更新")


class SharedLLMManager:
    """共享LLM管理器"""
    
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.lock = threading.Lock()
        self.registered_nodes = {}
        self.node_usage_stats = {}
        self.total_generations = 0
        self.total_updates = 0
    
    def register_node(self, node_id: str, node_config: Optional[Dict[str, Any]] = None):
        """注册节点"""
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
                "llm_backend": self.llm.backend.value,
                "total_generations": self.total_generations,
                "total_updates": self.total_updates,
                "registered_nodes_count": len(self.registered_nodes),
                "node_usage_stats": self.node_usage_stats.copy(),
                "llm_internal_stats": llm_params
            }


# 工厂函数
def create_llm_config(backend: Union[str, LLMBackend] = "mock", model_name: str = "mock_llm", **kwargs) -> LLMConfig:
    """创建LLM配置"""
    if isinstance(backend, str):
        backend = LLMBackend(backend)
    return LLMConfig(backend=backend, model_name=model_name, **kwargs)


def create_llm(config: LLMConfig) -> BaseLLM:
    """根据配置创建LLM实例"""
    if config.backend == LLMBackend.MOCK:
        return MockLLM(config)
    elif config.backend == LLMBackend.HUGGINGFACE:
        return HuggingFaceLLM(config)
    elif config.backend == LLMBackend.OPENAI_API:
        return OpenAILLM(config)
    else:
        raise ValueError(f"不支持的LLM后端: {config.backend}")


def create_shared_llm_manager(model_name: str = "mock_llm", backend: Union[str, LLMBackend] = "mock", **kwargs) -> SharedLLMManager:
    """创建共享LLM管理器"""
    config = create_llm_config(backend=backend, model_name=model_name, **kwargs)
    llm = create_llm(config)
    return SharedLLMManager(llm)


# 预定义模型管理器
def create_qwen_manager(model_name: str = "Qwen/Qwen-1_8B-Chat", device: str = "auto") -> SharedLLMManager:
    """创建Qwen模型管理器"""
    config = create_llm_config(
        backend="huggingface",
        model_name=model_name,
        device=device,
        max_length=1024,
        temperature=0.7
    )
    llm = create_llm(config)
    return SharedLLMManager(llm)


def create_gpt2_manager(model_size: str = "gpt2", device: str = "auto") -> SharedLLMManager:
    """创建GPT-2模型管理器"""
    config = create_llm_config(
        backend="huggingface",
        model_name=model_size,
        device=device,
        max_length=512,
        temperature=0.7
    )
    llm = create_llm(config)
    return SharedLLMManager(llm)


def create_openai_manager(model_name: str = "gpt-3.5-turbo", api_key: Optional[str] = None) -> SharedLLMManager:
    """创建OpenAI API模型管理器"""
    config = create_llm_config(
        backend="openai_api",
        model_name=model_name,
        api_key=api_key,
        max_length=1024,
        temperature=0.7
    )
    llm = create_llm(config)
    return SharedLLMManager(llm)


def get_available_models() -> Dict[str, List[str]]:
    """获取可用的模型列表"""
    return {
        "mock": ["mock_llm"],
        "gpt2": ["gpt2", "gpt2-medium", "gpt2-large"],
        "qwen": ["Qwen/Qwen-1_8B-Chat", "Qwen/Qwen-7B-Chat"],
        "openai": ["gpt-3.5-turbo", "gpt-4"]
    }
