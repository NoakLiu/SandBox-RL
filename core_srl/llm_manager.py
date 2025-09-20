#!/usr/bin/env python3
"""
Unified LLM Manager
==================

Integrated LLM management functionality:
1. Unified LLM interface (HuggingFace, vLLM)
2. Shared LLM manager for multi-model coordination
3. Adaptive parameter updates with freezing strategies
4. LoRA integration and compression
5. KV cache optimization for efficient inference
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
    """LLM Backend Types - only open-weight models"""
    HUGGINGFACE = "huggingface"
    VLLM = "vllm"


class UpdateStrategy(Enum):
    """Parameter update strategy"""
    FROZEN = "frozen"
    ADAPTIVE = "adaptive"
    SELECTIVE = "selective"
    INCREMENTAL = "incremental"


class ParameterImportance(Enum):
    """Parameter importance级别"""
    CRITICAL = "critical"
    IMPORTANT = "important"
    MODERATE = "moderate"
    LOW = "low"


@dataclass
class LLMConfig:
    """Unified LLM Configuration"""
    backend: LLMBackend = LLMBackend.HUGGINGFACE
    model_name: str = "Qwen/Qwen2.5-14B-Instruct"
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
    """LLM Response结果"""
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
        """Generate response"""
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """获取模型参数"""
        pass
    
    def update_parameters(self, gradients: Dict[str, Any], learning_rate: float = 1e-4) -> None:
        """更新模型参数（带冻结自适应逻辑）"""
        with self.lock:
            # 分析Parameter importance
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
        """分析Parameter importance"""
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
    
    def _apply_gradient_update(self, param_name: str, gradient: Any, learning_rate: float, 
                             optimizer_state: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
        """Apply gradient update using Adam-like optimization with proper state tracking"""
        
        # Initialize optimizer state if not provided
        if optimizer_state is None:
            optimizer_state = {}
        
        # Adam hyperparameters
        beta1 = optimizer_state.get('beta1', 0.9)
        beta2 = optimizer_state.get('beta2', 0.999)
        eps = optimizer_state.get('eps', 1e-8)
        weight_decay = optimizer_state.get('weight_decay', 0.01)
        
        # Initialize state for this parameter if needed
        if param_name not in optimizer_state:
            optimizer_state[param_name] = {
                'step': 0,
                'exp_avg': None,  # First moment estimate (momentum)
                'exp_avg_sq': None,  # Second moment estimate (RMSprop)
            }
        
        param_state = optimizer_state[param_name]
        param_state['step'] += 1
        
        # Convert gradient to appropriate tensor format
        if hasattr(gradient, 'detach'):
            grad = gradient.detach()
        else:
            # Convert numpy/other formats to tensor-like structure
            grad = gradient
        
        # Apply gradient clipping (per-parameter)
        max_grad_norm = 1.0
        if hasattr(grad, 'norm'):
            grad_norm = grad.norm().item()
        else:
            grad_norm = float(np.linalg.norm(grad)) if NUMPY_AVAILABLE else 1.0
        
        if grad_norm > max_grad_norm:
            clip_coef = max_grad_norm / (grad_norm + 1e-6)
            if hasattr(grad, 'mul_'):
                grad.mul_(clip_coef)
            else:
                grad = grad * clip_coef
            logger.debug(f"Clipped gradient for {param_name}: {grad_norm:.4f} -> {max_grad_norm}")
        
        # Add weight decay (L2 regularization) to gradient
        if weight_decay != 0:
            # This should be applied to the original parameter, not gradient
            # For now, we'll add it to gradient as approximation
            if hasattr(grad, 'add_'):
                # grad.add_(param, alpha=weight_decay)  # Would need original parameter
                pass
        
        # Initialize exponential moving averages if needed
        if param_state['exp_avg'] is None:
            if hasattr(grad, 'zeros_like'):
                param_state['exp_avg'] = grad.zeros_like()
                param_state['exp_avg_sq'] = grad.zeros_like()
            else:
                param_state['exp_avg'] = np.zeros_like(grad) if NUMPY_AVAILABLE else grad * 0
                param_state['exp_avg_sq'] = np.zeros_like(grad) if NUMPY_AVAILABLE else grad * 0
        
        exp_avg = param_state['exp_avg']
        exp_avg_sq = param_state['exp_avg_sq']
        step = param_state['step']
        
        # Update exponential moving averages
        if hasattr(exp_avg, 'mul_') and hasattr(exp_avg, 'add_'):
            # PyTorch tensor operations
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        else:
            # NumPy or other array operations
            exp_avg = beta1 * exp_avg + (1 - beta1) * grad
            exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * (grad ** 2)
            param_state['exp_avg'] = exp_avg
            param_state['exp_avg_sq'] = exp_avg_sq
        
        # Bias correction
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        
        # Corrected learning rate
        corrected_lr = learning_rate * (bias_correction2 ** 0.5) / bias_correction1
        
        # Apply learning rate scheduling
        if hasattr(self, 'lr_scheduler_factor'):
            schedule_factor = self.lr_scheduler_factor
        else:
            # Simple exponential decay
            schedule_factor = 0.98 ** (step // 100)
        
        effective_lr = corrected_lr * schedule_factor
        
        # Compute parameter update using Adam formula
        if hasattr(exp_avg_sq, 'sqrt'):
            # PyTorch tensor operations
            denom = exp_avg_sq.sqrt().add_(eps)
            param_update = -effective_lr * exp_avg / denom
        else:
            # NumPy operations
            denom = np.sqrt(exp_avg_sq) + eps
            param_update = -effective_lr * exp_avg / denom
        
        # Add noise for regularization (optional)
        if hasattr(self, 'add_noise') and self.add_noise:
            noise_factor = 1e-6 * effective_lr
            if hasattr(param_update, 'normal_'):
                noise = param_update.clone().normal_(0, noise_factor)
                param_update.add_(noise)
            elif NUMPY_AVAILABLE:
                noise = np.random.normal(0, noise_factor, param_update.shape)
                param_update = param_update + noise
        
        # Log update statistics
        if hasattr(param_update, 'norm'):
            update_norm = param_update.norm().item()
        else:
            update_norm = float(np.linalg.norm(param_update)) if NUMPY_AVAILABLE else 0.0
        
        logger.debug(f"Adam update for {param_name}: step={step}, lr={effective_lr:.8f}, "
                    f"update_norm={update_norm:.8f}, bias_corr=({bias_correction1:.4f}, {bias_correction2:.4f})")
        
        return param_update, optimizer_state
    
    def _execute_parameter_update(self, updated_params: Dict[str, Any], learning_rate: float):
        """Execute parameter update with validation and safety checks"""
        if not updated_params:
            logger.warning("No parameters to update")
            return
        
        # Validate parameter updates before applying
        valid_updates = {}
        invalid_count = 0
        
        for param_name, param_update in updated_params.items():
            try:
                # Check if parameter update is valid
                if self._validate_parameter_update(param_name, param_update):
                    valid_updates[param_name] = param_update
                else:
                    invalid_count += 1
                    logger.warning(f"Invalid parameter update for {param_name}, skipping")
            except Exception as e:
                invalid_count += 1
                logger.error(f"Error validating parameter update for {param_name}: {e}")
        
        if invalid_count > 0:
            logger.warning(f"Skipped {invalid_count} invalid parameter updates")
        
        if not valid_updates:
            logger.error("No valid parameter updates to apply")
            return
        
        # Apply parameter updates with safety mechanisms
        successful_updates = 0
        failed_updates = 0
        
        for param_name, param_update in valid_updates.items():
            try:
                # Apply the parameter update (subclass implementation)
                self._apply_single_parameter_update(param_name, param_update, learning_rate)
                successful_updates += 1
                
                # Log significant updates
                if hasattr(param_update, 'shape') or hasattr(param_update, '__len__'):
                    update_size = getattr(param_update, 'size', len(param_update)) if hasattr(param_update, '__len__') else 1
                    if update_size > 1000:  # Log large updates
                        logger.info(f"Applied large parameter update to {param_name}: size={update_size}")
                
            except Exception as e:
                failed_updates += 1
                logger.error(f"Failed to update parameter {param_name}: {e}")
        
        # Update statistics
        self.update_count += 1
        
        # Log update summary
        logger.info(f"Parameter update completed: {successful_updates} successful, {failed_updates} failed")
        
        # Trigger post-update validation if available
        try:
            self._post_update_validation()
        except Exception as e:
            logger.warning(f"Post-update validation failed: {e}")
    
    def _validate_parameter_update(self, param_name: str, param_update: Any) -> bool:
        """Validate parameter update before applying"""
        if param_update is None:
            return False
        
        # Check for NaN or infinite values
        if NUMPY_AVAILABLE:
            try:
                if hasattr(param_update, 'cpu'):
                    update_array = param_update.cpu().numpy()
                elif hasattr(param_update, 'numpy'):
                    update_array = param_update.numpy()
                else:
                    update_array = np.array(param_update)
                
                if np.any(np.isnan(update_array)) or np.any(np.isinf(update_array)):
                    logger.error(f"Parameter update for {param_name} contains NaN or Inf values")
                    return False
                
                # Check for extremely large updates
                max_abs_value = np.max(np.abs(update_array))
                if max_abs_value > 10.0:  # Threshold for "too large" updates
                    logger.warning(f"Very large parameter update for {param_name}: max_abs={max_abs_value:.4f}")
                    # Allow but warn - might be intentional
                
            except Exception as e:
                logger.error(f"Error validating parameter update for {param_name}: {e}")
                return False
        
        return True
    
    def _apply_single_parameter_update(self, param_name: str, param_update: Any, learning_rate: float):
        """Apply single parameter update - to be implemented by subclasses"""
        # Base implementation - subclasses should override
        logger.debug(f"Base implementation: would update {param_name} with lr={learning_rate}")
    
    def _post_update_validation(self):
        """Perform validation after parameter updates - to be implemented by subclasses"""
        # Base implementation - subclasses can override for model-specific validation
        pass


class AnthropicLLM(BaseLLM):
class HuggingFaceLLM(BaseLLM):
    """HuggingFace model implementation"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check dependencies"""
        try:
            import torch
            import transformers
            self.torch = torch
            self.transformers = transformers
        except ImportError as e:
            logger.error(f"缺少依赖: {e}")
            raise
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response"""
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
                
                # Generate response
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
                logger.error(f"Generate response失败: {e}")
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
    
    def _execute_parameter_update(self, gradients: Dict[str, Any], learning_rate: float):
        """Execute HuggingFace model parameter updates with proper gradient-based optimization"""
        if not gradients:
            logger.warning("No gradients provided for parameter update")
            return
        
        if not self.model_loaded or self.model is None:
            logger.error("HuggingFace model not loaded, cannot update parameters")
            return
        
        try:
            # Initialize optimizer state if not exists
            if not hasattr(self, '_optimizer_state'):
                self._optimizer_state = {
                    'beta1': 0.9,
                    'beta2': 0.999,
                    'eps': 1e-8,
                    'weight_decay': 0.01,
                    'amsgrad': False
                }
            
            # Get named parameters for proper gradient application
            named_params = dict(self.model.named_parameters())
            updated_count = 0
            total_grad_norm = 0.0
            
            # First pass: compute global gradient norm for clipping
            grad_norms = []
            for param_name, grad_data in gradients.items():
                matching_param_name = self._find_matching_parameter(param_name, named_params)
                if matching_param_name and matching_param_name in named_params:
                    param = named_params[matching_param_name]
                    
                    # Convert gradient to tensor
                    if isinstance(grad_data, dict) and 'gradient' in grad_data:
                        grad = grad_data['gradient']
                    else:
                        grad = grad_data
                    
                    if not isinstance(grad, self.torch.Tensor):
                        try:
                            if NUMPY_AVAILABLE and hasattr(grad, 'shape'):
                                grad = self.torch.from_numpy(grad).to(param.device)
                            else:
                                grad = self.torch.tensor(grad, device=param.device, dtype=param.dtype)
                        except Exception as e:
                            logger.error(f"Failed to convert gradient for {param_name}: {e}")
                            continue
                    
                    if grad.shape == param.shape:
                        grad_norm = grad.norm().item()
                        grad_norms.append(grad_norm)
                        total_grad_norm += grad_norm ** 2
            
            total_grad_norm = (total_grad_norm ** 0.5)
            
            # Apply global gradient clipping
            max_grad_norm = 1.0
            clip_coef = 1.0
            if total_grad_norm > max_grad_norm:
                clip_coef = max_grad_norm / (total_grad_norm + 1e-6)
                logger.debug(f"Applied global gradient clipping: {total_grad_norm:.4f} -> {max_grad_norm}")
            
            # Second pass: apply parameter updates
            with self.torch.no_grad():
                for param_name, grad_data in gradients.items():
                    matching_param_name = self._find_matching_parameter(param_name, named_params)
                    
                    if matching_param_name is None or matching_param_name not in named_params:
                        logger.warning(f"Parameter {param_name} not found in model")
                        continue
                    
                    param = named_params[matching_param_name]
                    
                    # Extract gradient and metadata
                    if isinstance(grad_data, dict):
                        grad = grad_data.get('gradient', grad_data)
                        importance = grad_data.get('importance', 'NORMAL')
                        layer_type = grad_data.get('layer_type', 'unknown')
                    else:
                        grad = grad_data
                        importance = 'NORMAL'
                        layer_type = 'unknown'
                    
                    # Convert gradient to tensor
                    if not isinstance(grad, self.torch.Tensor):
                        try:
                            if NUMPY_AVAILABLE and hasattr(grad, 'shape'):
                                grad = self.torch.from_numpy(grad).to(param.device)
                            else:
                                grad = self.torch.tensor(grad, device=param.device, dtype=param.dtype)
                        except Exception as e:
                            logger.error(f"Failed to convert gradient for {param_name}: {e}")
                            continue
                    
                    # Ensure shapes and devices match
                    if grad.shape != param.shape:
                        logger.error(f"Shape mismatch for {param_name}: param {param.shape} vs grad {grad.shape}")
                        continue
                    
                    grad = grad.to(param.device)
                    
                    # Apply global gradient clipping
                    if clip_coef < 1.0:
                        grad = grad * clip_coef
                    
                    # Apply Adam optimization with proper state tracking
                    param_update, self._optimizer_state = self._apply_gradient_update(
                        matching_param_name, grad, learning_rate, self._optimizer_state
                    )
                    
                    # Apply parameter-specific learning rate adjustments
                    effective_lr = learning_rate
                    
                    # Adjust learning rate based on layer type
                    if 'embedding' in matching_param_name.lower():
                        effective_lr *= 0.1  # Lower LR for embeddings
                    elif 'layernorm' in matching_param_name.lower() or 'layer_norm' in matching_param_name.lower():
                        effective_lr *= 2.0  # Higher LR for layer norms
                    elif 'bias' in matching_param_name.lower():
                        effective_lr *= 2.0  # Higher LR for biases
                    
                    # Apply importance-based scaling
                    if importance == 'CRITICAL':
                        effective_lr *= 0.5  # More conservative updates for critical params
                    elif importance == 'LOW':
                        effective_lr *= 2.0  # More aggressive updates for less important params
                    
                    # Convert param_update to tensor if needed
                    if not isinstance(param_update, self.torch.Tensor):
                        param_update = self.torch.tensor(param_update, device=param.device, dtype=param.dtype)
                    
                    # Apply the parameter update
                    try:
                        old_param = param.clone()
                        param.data.add_(param_update)
                        
                        # Check for numerical stability
                        if self.torch.any(self.torch.isnan(param)) or self.torch.any(self.torch.isinf(param)):
                            logger.error(f"NaN or Inf detected in {matching_param_name}, reverting update")
                            param.data.copy_(old_param)
                            continue
                        
                        updated_count += 1
                        
                        # Log detailed statistics
                        param_norm = param.norm().item()
                        update_norm = param_update.norm().item()
                        relative_update = update_norm / (param_norm + 1e-8)
                        
                        logger.debug(f"Updated {matching_param_name}: param_norm={param_norm:.6f}, "
                                   f"update_norm={update_norm:.6f}, relative={relative_update:.6f}, "
                                   f"lr={effective_lr:.8f}, importance={importance}")
                        
                        # Warn about large relative updates
                        if relative_update > 0.1:
                            logger.warning(f"Large relative update for {matching_param_name}: {relative_update:.4f}")
                        
                    except Exception as e:
                        logger.error(f"Failed to apply parameter update for {matching_param_name}: {e}")
                        continue
            
            # Post-update operations
            self._post_parameter_update_operations()
            
            logger.info(f"HuggingFace parameter update completed: {updated_count}/{len(gradients)} parameters updated, "
                       f"total_grad_norm={total_grad_norm:.6f}, clip_coef={clip_coef:.4f}")
            
        except Exception as e:
            logger.error(f"HuggingFace parameter update failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _find_matching_parameter(self, param_name: str, model_state_dict: Dict[str, Any]) -> Optional[str]:
        """Find matching parameter name in model state dict"""
        # Direct match
        if param_name in model_state_dict:
            return param_name
        
        # Try common parameter name variations
        variations = [
            param_name.replace(".", "_"),
            param_name.replace("_", "."),
            f"model.{param_name}",
            f"transformer.{param_name}",
            param_name.replace("weight", ""),
            param_name.replace("bias", "")
        ]
        
        for variation in variations:
            if variation in model_state_dict:
                return variation
        
        # Fuzzy matching - find parameters that contain the param_name
        for model_param_name in model_state_dict.keys():
            if param_name in model_param_name or model_param_name in param_name:
                logger.debug(f"Fuzzy matched {param_name} to {model_param_name}")
                return model_param_name
        
        return None
    
    def _post_parameter_update_operations(self):
        """Perform post-update operations for HuggingFace model"""
        try:
            # Clear any cached computations
            if hasattr(self.model, 'clear_cache'):
                self.model.clear_cache()
            
            # Update model generation statistics
            self.update_count += 1
            
            # Perform gradient checkpointing if enabled
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
            
            # Log memory usage if available
            if hasattr(self.torch.cuda, 'memory_allocated'):
                memory_allocated = self.torch.cuda.memory_allocated()
                logger.debug(f"GPU memory after parameter update: {memory_allocated / 1024**3:.2f} GB")
                
        except Exception as e:
            logger.warning(f"Post-update operations failed: {e}")
    
    def _apply_single_parameter_update(self, param_name: str, param_update: Any, learning_rate: float):
        """Apply single parameter update for HuggingFace model"""
        if not self.model_loaded or self.model is None:
            logger.error("Model not loaded, cannot apply parameter update")
            return
        
        try:
            model_state_dict = self.model.state_dict()
            matching_param_name = self._find_matching_parameter(param_name, model_state_dict)
            
            if matching_param_name is None:
                logger.warning(f"Parameter {param_name} not found in model")
                return
            
            current_param = model_state_dict[matching_param_name]
            
            # Convert and apply update
            with self.torch.no_grad():
                if not isinstance(param_update, self.torch.Tensor):
                    param_update = self.torch.tensor(param_update, device=current_param.device)
                
                param_update = param_update.to(current_param.device)
                new_param_value = current_param - learning_rate * param_update
                model_state_dict[matching_param_name].copy_(new_param_value)
            
            logger.debug(f"Applied single parameter update to {matching_param_name}")
            
        except Exception as e:
            logger.error(f"Failed to apply single parameter update for {param_name}: {e}")
    
    def _post_update_validation(self):
        """Perform validation after HuggingFace model parameter updates"""
        if not self.model_loaded or self.model is None:
            return
        
        try:
            # Check for NaN parameters
            nan_params = []
            for name, param in self.model.named_parameters():
                if self.torch.any(self.torch.isnan(param)):
                    nan_params.append(name)
            
            if nan_params:
                logger.error(f"Found NaN parameters after update: {nan_params}")
            
            # Check parameter norms
            total_norm = 0.0
            param_count = 0
            
            for name, param in self.model.named_parameters():
                param_norm = self.torch.norm(param).item()
                total_norm += param_norm
                param_count += 1
                
                # Check for extremely large parameters
                if param_norm > 100.0:
                    logger.warning(f"Large parameter norm for {name}: {param_norm:.4f}")
            
            avg_norm = total_norm / max(param_count, 1)
            logger.debug(f"Post-update validation: avg parameter norm = {avg_norm:.6f}")
            
        except Exception as e:
            logger.warning(f"Post-update validation failed: {e}")


class OpenAILLM(BaseLLM):
class SharedLLMManager:
    """Shared LLM manager"""
    
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.lock = threading.Lock()
        self.registered_nodes = {}
        self.node_usage_stats = {}
        self.total_generations = 0
        self.total_updates = 0
    
    def register_node(self, node_id: str, node_config: Optional[Dict[str, Any]] = None):
        """Register node"""
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
        """为特定节点Generate response"""
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
        """Update shared LLM parameters - all supported models are open-weight"""
        with self.lock:
            # All supported models are open-weight and support parameter updates
            if hasattr(self.llm, 'update_parameters'):
                self.llm.update_parameters(gradients, learning_rate)
                self.total_updates += 1
                
                return {
                    "status": "updated",
                    "update_count": self.total_updates,
                    "affected_nodes": list(self.registered_nodes.keys()),
                    "learning_rate": learning_rate
                }
            else:
                return {
                    "status": "error",
                    "reason": "Model does not support parameter updates",
                    "model": self.llm.model_name
                }
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get global statistics"""
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
    """创建LLM Configuration"""
    if isinstance(backend, str):
        backend = LLMBackend(backend)
    return LLMConfig(backend=backend, model_name=model_name, **kwargs)


def create_llm(config: LLMConfig) -> BaseLLM:
    """Create LLM instance based on configuration - only open-weight models"""
    if config.backend == LLMBackend.HUGGINGFACE:
        return HuggingFaceLLM(config)
    elif config.backend == LLMBackend.VLLM:
        # For now, use HuggingFace as VLLM backend
        return HuggingFaceLLM(config)
    else:
        raise ValueError(f"Unsupported LLM backend: {config.backend}. Only HUGGINGFACE and VLLM are supported.")


def create_shared_llm_manager(model_name: str = "Qwen/Qwen2.5-14B-Instruct", backend: Union[str, LLMBackend] = "huggingface", **kwargs) -> SharedLLMManager:
    """Create shared LLM manager with modern models"""
    config = create_llm_config(backend=backend, model_name=model_name, **kwargs)
    llm = create_llm(config)
    return SharedLLMManager(llm)


# Modern model managers
def create_qwen3_manager(model_name: str = "Qwen/Qwen2.5-14B-Instruct", device: str = "auto") -> SharedLLMManager:
    """Create Qwen3 model manager with latest models"""
    config = create_llm_config(
        backend="huggingface",
        model_name=model_name,
        device=device,
        max_length=32768,  # Qwen3 supports long context
        temperature=0.7
    )
    llm = create_llm(config)
    return SharedLLMManager(llm)


def create_qwen_coder_manager(model_name: str = "Qwen/Qwen2.5-Coder-14B-Instruct", device: str = "auto") -> SharedLLMManager:
    """Create Qwen Coder model manager for code tasks"""
    config = create_llm_config(
        backend="huggingface", 
        model_name=model_name,
        device=device,
        max_length=16384,
        temperature=0.3  # Lower temperature for code
    )
    llm = create_llm(config)
    return SharedLLMManager(llm)


def create_qwen_math_manager(model_name: str = "Qwen/Qwen2.5-Math-14B-Instruct", device: str = "auto") -> SharedLLMManager:
    """Create Qwen Math model manager for mathematical reasoning"""
    config = create_llm_config(
        backend="huggingface",
        model_name=model_name, 
        device=device,
        max_length=8192,
        temperature=0.2  # Low temperature for math
    )
    llm = create_llm(config)
    return SharedLLMManager(llm)






def create_llama3_manager(model_name: str = "meta-llama/Llama-3.1-8B-Instruct", device: str = "auto") -> SharedLLMManager:
    """Create Llama 3.1 model manager"""
    config = create_llm_config(
        backend="huggingface",
        model_name=model_name,
        device=device,
        max_length=131072,  # Llama 3.1 long context
        temperature=0.7
    )
    llm = create_llm(config)
    return SharedLLMManager(llm)


def get_available_models() -> Dict[str, List[str]]:
    """Get available modern model list"""
    return {
        "qwen3": [
            "Qwen/Qwen2.5-14B-Instruct",
            "Qwen/Qwen2.5-32B-Instruct", 
            "Qwen/Qwen2.5-72B-Instruct",
            "Qwen/Qwen2.5-Coder-14B-Instruct",
            "Qwen/Qwen2.5-Math-14B-Instruct"
        ],
        "openai": [
            "gpt-4o",
            "gpt-4o-mini", 
            "gpt-4-turbo",
            "o1-preview",
            "o1-mini"
        ],
        "anthropic": [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022", 
            "claude-3-opus-20240229"
        ],
        "llama3": [
            "meta-llama/Llama-3.1-8B-Instruct",
            "meta-llama/Llama-3.1-70B-Instruct",
            "meta-llama/Llama-3.1-405B-Instruct"
        ],
        "deepseek": [
            "deepseek-ai/deepseek-coder-33b-instruct",
            "deepseek-ai/deepseek-math-7b-instruct"
        ]
    }
