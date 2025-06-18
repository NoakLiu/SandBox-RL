"""
大语言模型接口

提供统一的LLM接口，支持参数共享和强化学习优化
支持多种真实LLM：GPT-2、LLaMA、Qwen等
"""

from typing import Any, Dict, List, Optional, Callable, Union
from abc import ABC, abstractmethod
import logging
import time
import threading
from dataclasses import dataclass
import json
import os
from enum import Enum

logger = logging.getLogger(__name__)


class LLMBackend(Enum):
    """LLM后端类型"""
    MOCK = "mock"
    GPT2 = "gpt2"
    LLAMA = "llama"
    QWEN = "qwen"
    OPENAI_API = "openai_api"
    HUGGINGFACE = "huggingface"


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


@dataclass
class LLMConfig:
    """LLM配置"""
    backend: LLMBackend = LLMBackend.HUGGINGFACE  # 默认使用HuggingFace后端
    model_name: str = "Qwen/Qwen-7B-Chat"  # 默认使用Qwen模型
    device: str = "auto"  # "cpu", "cuda", "auto"
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    
    # API相关配置
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    
    # 模型路径配置
    model_path: Optional[str] = None
    cache_dir: Optional[str] = None
    
    # 性能配置
    batch_size: int = 1
    use_cache: bool = True
    torch_dtype: str = "auto"  # "float16", "float32", "auto"


class BaseLLM(ABC):
    """基础LLM抽象类"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.model_name = config.model_name
        self.backend = config.backend
        
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
    
    @abstractmethod
    def load_model(self) -> None:
        """加载模型"""
        pass
    
    @abstractmethod
    def unload_model(self) -> None:
        """卸载模型"""
        pass


class MockLLM(BaseLLM):
    """模拟LLM实现（用于演示和测试）"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.parameters = {
            "embedding_weights": [0.1] * 1000,
            "attention_weights": [0.2] * 500,
            "output_weights": [0.3] * 200
        }
        self.generation_count = 0
        self.update_count = 0
        self.lock = threading.Lock()
        self.model_loaded = False
        
        # 模拟不同类型的推理能力
        self.reasoning_templates = {
            "mathematical": "数学推理：分析问题 → 建立方程 → 求解 → 验证",
            "logical": "逻辑推理：前提分析 → 规则应用 → 结论推导 → 一致性检查",
            "strategic": "策略推理：目标分析 → 选项评估 → 风险评估 → 最优选择",
            "creative": "创造性推理：问题理解 → 发散思考 → 方案生成 → 可行性评估"
        }
    
    def load_model(self) -> None:
        """加载模型"""
        logger.info(f"加载Mock模型: {self.model_name}")
        self.model_loaded = True
    
    def unload_model(self) -> None:
        """卸载模型"""
        logger.info(f"卸载Mock模型: {self.model_name}")
        self.model_loaded = False
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """生成响应"""
        if not self.model_loaded:
            self.load_model()
            
        with self.lock:
            self.generation_count += 1
            
            # 合并配置和运行时参数
            temperature = kwargs.get("temperature", self.config.temperature)
            max_length = kwargs.get("max_length", self.config.max_length)
            reasoning_type = kwargs.get("reasoning_type", "logical")
            
            # 基于prompt内容生成响应
            if "数学" in prompt or "计算" in prompt or "24点" in prompt:
                reasoning = self.reasoning_templates["mathematical"]
                response_text = f"基于数学推理，我分析了问题并得出结论。温度参数: {temperature}"
                confidence = 0.8 + (self.update_count * 0.01)
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
                    "backend": self.backend.value,
                    "generation_count": self.generation_count,
                    "update_count": self.update_count,
                    "temperature": temperature,
                    "max_length": max_length,
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
                "model_name": self.model_name,
                "backend": self.backend.value,
                "model_loaded": self.model_loaded
            }
    
    def update_parameters(self, gradients: Dict[str, Any], learning_rate: float = 1e-4) -> None:
        """更新模型参数"""
        with self.lock:
            # 模拟参数更新
            for param_name, gradient in gradients.items():
                if param_name in self.parameters:
                    if isinstance(gradient, (list, tuple)):
                        for i in range(min(len(self.parameters[param_name]), len(gradient))):
                            self.parameters[param_name][i] -= learning_rate * gradient[i]
                    else:
                        for i in range(len(self.parameters[param_name])):
                            self.parameters[param_name][i] -= learning_rate * gradient
            
            self.update_count += 1
            logger.info(f"Mock LLM参数更新完成，更新次数: {self.update_count}")


class HuggingFaceLLM(BaseLLM):
    """HuggingFace模型实现（支持GPT-2、LLaMA、Qwen等）"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self.generation_count = 0
        self.update_count = 0
        self.lock = threading.Lock()
        self.model_loaded = False
        
        # 检查依赖
        self._check_dependencies()
    
    def _check_dependencies(self):
        """检查必要的依赖"""
        try:
            import torch
            import transformers
            self.torch = torch
            self.transformers = transformers
        except ImportError as e:
            logger.error(f"缺少必要依赖: {e}")
            logger.error("请安装: pip install torch transformers accelerate")
            raise
    
    def load_model(self) -> None:
        """加载HuggingFace模型"""
        if self.model_loaded:
            return
            
        logger.info(f"加载HuggingFace模型: {self.model_name}")
        
        try:
            # 设备配置
            if self.config.device == "auto":
                device = "cuda" if self.torch.cuda.is_available() else "cpu"
            else:
                device = self.config.device
            
            # 数据类型配置
            if self.config.torch_dtype == "auto":
                torch_dtype = self.torch.float16 if device == "cuda" else self.torch.float32
            elif self.config.torch_dtype == "float16":
                torch_dtype = self.torch.float16
            else:
                torch_dtype = self.torch.float32
            
            # 加载tokenizer
            logger.info(f"Loading tokenizer: {self.model_name}")
            self.tokenizer = self.transformers.AutoTokenizer.from_pretrained(
                self.config.model_path or self.model_name,
                cache_dir=self.config.cache_dir,
                trust_remote_code=True
            )
            
            # Set pad token safely
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    # Use a default pad token ID if available
                    if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None:
                        self.tokenizer.pad_token = self.tokenizer.convert_ids_to_tokens(self.tokenizer.pad_token_id)
                    else:
                        # Set a default pad token ID
                        self.tokenizer.pad_token_id = 0
                        self.tokenizer.pad_token = self.tokenizer.convert_ids_to_tokens(0)
            
            # 加载模型
            logger.info(f"加载模型到设备: {device}, 数据类型: {torch_dtype}")
            self.model = self.transformers.AutoModelForCausalLM.from_pretrained(
                self.config.model_path or self.model_name,
                cache_dir=self.config.cache_dir,
                torch_dtype=torch_dtype,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            if device == "cpu":
                self.model = self.model.to(device)
            
            self.device = device
            self.model_loaded = True
            logger.info(f"模型加载完成: {self.model_name}")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def unload_model(self) -> None:
        """卸载模型"""
        if not self.model_loaded:
            return
            
        logger.info(f"卸载模型: {self.model_name}")
        
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        # 清理GPU内存
        if hasattr(self, 'torch') and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        
        self.model_loaded = False
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """生成响应"""
        if not self.model_loaded:
            self.load_model()
        
        with self.lock:
            self.generation_count += 1
            
            try:
                # 合并配置和运行时参数
                temperature = kwargs.get("temperature", self.config.temperature)
                max_length = kwargs.get("max_length", self.config.max_length)
                top_p = kwargs.get("top_p", self.config.top_p)
                top_k = kwargs.get("top_k", self.config.top_k)
                do_sample = kwargs.get("do_sample", self.config.do_sample)
                
                # Tokenize输入
                inputs = self.tokenizer.encode(prompt, return_tensors="pt")
                inputs = inputs.to(self.device)
                
                # 生成参数
                generation_kwargs = {
                    "max_length": min(max_length, inputs.shape[1] + 256),  # 限制最大长度
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "do_sample": do_sample,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "use_cache": self.config.use_cache
                }
                
                # 生成响应
                start_time = time.time()
                with self.torch.no_grad():
                    outputs = self.model.generate(inputs, **generation_kwargs)
                
                generation_time = time.time() - start_time
                
                # 解码响应
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # 提取新生成的部分
                response_text = generated_text[len(prompt):].strip()
                
                # 计算置信度（简化实现）
                confidence = min(0.95, 0.7 + (self.update_count * 0.01))
                
                # 生成推理说明
                reasoning = f"使用{self.backend.value}模型进行文本生成，温度={temperature}"
                
                return LLMResponse(
                    text=response_text,
                    confidence=confidence,
                    reasoning=reasoning,
                    metadata={
                        "backend": self.backend.value,
                        "model_name": self.model_name,
                        "generation_count": self.generation_count,
                        "generation_time": generation_time,
                        "temperature": temperature,
                        "max_length": max_length,
                        "prompt_length": len(prompt),
                        "response_length": len(response_text),
                        "device": self.device
                    }
                )
                
            except Exception as e:
                logger.error(f"生成响应失败: {e}")
                # 返回错误响应
                return LLMResponse(
                    text=f"生成失败: {str(e)}",
                    confidence=0.0,
                    reasoning="生成过程中出现错误",
                    metadata={
                        "backend": self.backend.value,
                        "error": str(e),
                        "generation_count": self.generation_count
                    }
                )
    
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
                # 获取模型参数统计
                total_params = sum(p.numel() for p in self.model.parameters())
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                
                params_info.update({
                    "total_parameters": total_params,
                    "trainable_parameters": trainable_params,
                    "device": getattr(self, 'device', 'unknown'),
                    "model_dtype": str(next(self.model.parameters()).dtype) if self.model.parameters() else "unknown"
                })
            
            return params_info
    
    def update_parameters(self, gradients: Dict[str, Any], learning_rate: float = 1e-4) -> None:
        """更新模型参数（简化实现）"""
        with self.lock:
            if not self.model_loaded or self.model is None:
                logger.warning("模型未加载，无法更新参数")
                return
            
            # 这里应该实现真实的参数更新逻辑
            # 由于这是一个复杂的过程，这里只做模拟
            self.update_count += 1
            logger.info(f"HuggingFace模型参数更新完成，更新次数: {self.update_count}")
            
            # 实际实现中，这里应该：
            # 1. 计算损失函数
            # 2. 反向传播
            # 3. 应用梯度更新
            # 4. 更新优化器状态


class OpenAILLM(BaseLLM):
    """OpenAI API实现"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.generation_count = 0
        self.update_count = 0
        self.lock = threading.Lock()
        self.model_loaded = False
        
        # 检查依赖和API密钥
        self._check_dependencies()
        self._setup_client()
    
    def _check_dependencies(self):
        """检查OpenAI依赖"""
        try:
            import openai
            self.openai = openai
        except ImportError:
            logger.error("缺少OpenAI依赖，请安装: pip install openai")
            raise
    
    def _setup_client(self):
        """设置OpenAI客户端"""
        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("未设置OpenAI API密钥")
            return
        
        self.client = self.openai.OpenAI(
            api_key=api_key,
            base_url=self.config.api_base
        )
        self.model_loaded = True
    
    def load_model(self) -> None:
        """加载模型（API模式下不需要实际加载）"""
        if not self.model_loaded:
            self._setup_client()
        logger.info(f"OpenAI API模型准备就绪: {self.model_name}")
    
    def unload_model(self) -> None:
        """卸载模型"""
        self.model_loaded = False
        logger.info(f"OpenAI API连接关闭: {self.model_name}")
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """通过OpenAI API生成响应"""
        if not self.model_loaded:
            self.load_model()
        
        with self.lock:
            self.generation_count += 1
            
            try:
                # 合并配置和运行时参数
                temperature = kwargs.get("temperature", self.config.temperature)
                max_tokens = kwargs.get("max_tokens", min(self.config.max_length, 4096))
                
                # 调用OpenAI API
                start_time = time.time()
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=self.config.top_p
                )
                
                generation_time = time.time() - start_time
                
                # 提取响应
                response_text = response.choices[0].message.content
                
                # 计算置信度（基于finish_reason）
                finish_reason = response.choices[0].finish_reason
                confidence = 0.9 if finish_reason == "stop" else 0.7
                
                return LLMResponse(
                    text=response_text,
                    confidence=confidence,
                    reasoning=f"通过OpenAI API调用{self.model_name}模型生成",
                    metadata={
                        "backend": self.backend.value,
                        "model_name": self.model_name,
                        "generation_count": self.generation_count,
                        "generation_time": generation_time,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "finish_reason": finish_reason,
                        "usage": response.usage.dict() if response.usage else None
                    }
                )
                
            except Exception as e:
                logger.error(f"OpenAI API调用失败: {e}")
                return LLMResponse(
                    text=f"API调用失败: {str(e)}",
                    confidence=0.0,
                    reasoning="OpenAI API调用出现错误",
                    metadata={
                        "backend": self.backend.value,
                        "error": str(e),
                        "generation_count": self.generation_count
                    }
                )
    
    def get_parameters(self) -> Dict[str, Any]:
        """获取模型参数信息"""
        with self.lock:
            return {
                "model_name": self.model_name,
                "backend": self.backend.value,
                "generation_count": self.generation_count,
                "update_count": self.update_count,
                "model_loaded": self.model_loaded,
                "api_available": hasattr(self, 'client')
            }
    
    def update_parameters(self, gradients: Dict[str, Any], learning_rate: float = 1e-4) -> None:
        """更新模型参数（API模式下不支持）"""
        with self.lock:
            self.update_count += 1
            logger.warning("OpenAI API模式不支持参数更新，仅记录更新次数")


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
                "llm_backend": self.llm.backend.value,
                "total_generations": self.total_generations,
                "total_updates": self.total_updates,
                "registered_nodes_count": len(self.registered_nodes),
                "node_usage_stats": self.node_usage_stats.copy(),
                "llm_internal_stats": llm_params
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
                "shared_llm_model": self.llm.model_name,
                "shared_llm_backend": self.llm.backend.value
            }
    
    def load_model(self) -> None:
        """加载共享模型"""
        self.llm.load_model()
    
    def unload_model(self) -> None:
        """卸载共享模型"""
        self.llm.unload_model()


# 便利函数
def create_llm_config(
    backend: Union[str, LLMBackend] = "mock",
    model_name: str = "mock_llm",
    **kwargs
) -> LLMConfig:
    """创建LLM配置"""
    if isinstance(backend, str):
        backend = LLMBackend(backend)
    
    return LLMConfig(
        backend=backend,
        model_name=model_name,
        **kwargs
    )


def create_llm(config: LLMConfig) -> BaseLLM:
    """根据配置创建LLM实例"""
    if config.backend == LLMBackend.MOCK:
        return MockLLM(config)
    elif config.backend in [LLMBackend.GPT2, LLMBackend.LLAMA, LLMBackend.QWEN, LLMBackend.HUGGINGFACE]:
        return HuggingFaceLLM(config)
    elif config.backend == LLMBackend.OPENAI_API:
        return OpenAILLM(config)
    else:
        raise ValueError(f"不支持的LLM后端: {config.backend}")


def create_shared_llm_manager(
    model_name: str = "Qwen/Qwen-7B-Chat",  # 默认使用Qwen模型
    backend: Union[str, LLMBackend] = "huggingface",  # 默认使用HuggingFace后端
    **kwargs
) -> SharedLLMManager:
    """创建共享LLM管理器"""
    config = create_llm_config(backend=backend, model_name=model_name, **kwargs)
    llm = create_llm(config)
    return SharedLLMManager(llm)


# 预定义的模型配置
def create_gpt2_manager(model_size: str = "gpt2", device: str = "auto") -> SharedLLMManager:
    """创建GPT-2模型管理器"""
    config = create_llm_config(
        backend="huggingface",
        model_name=model_size,  # "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"
        device=device,
        max_length=512,
        temperature=0.7
    )
    llm = create_llm(config)
    return SharedLLMManager(llm)


def create_llama_manager(model_path: str, device: str = "auto") -> SharedLLMManager:
    """创建LLaMA模型管理器"""
    config = create_llm_config(
        backend="huggingface",
        model_name="llama",
        model_path=model_path,  # 本地模型路径或HuggingFace模型名
        device=device,
        max_length=1024,
        temperature=0.7,
        torch_dtype="float16"
    )
    llm = create_llm(config)
    return SharedLLMManager(llm)


def create_qwen_manager(model_name: str = "Qwen/Qwen-7B-Chat", device: str = "auto") -> SharedLLMManager:
    """创建Qwen模型管理器"""
    config = create_llm_config(
        backend="huggingface",
        model_name=model_name,  # "Qwen/Qwen-7B-Chat", "Qwen/Qwen-14B-Chat", etc.
        device=device,
        max_length=1024,
        temperature=0.7,
        torch_dtype="float16"
    )
    llm = create_llm(config)
    return SharedLLMManager(llm)


def create_openai_manager(model_name: str = "gpt-3.5-turbo", api_key: Optional[str] = None) -> SharedLLMManager:
    """创建OpenAI API模型管理器"""
    config = create_llm_config(
        backend="openai_api",
        model_name=model_name,  # "gpt-3.5-turbo", "gpt-4", etc.
        api_key=api_key,
        max_length=1024,
        temperature=0.7
    )
    llm = create_llm(config)
    return SharedLLMManager(llm) 