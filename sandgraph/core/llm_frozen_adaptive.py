"""
LLMs Frozen & Adaptive Update Module
====================================

提供大语言模型的冻结和自适应更新功能，支持：
- 模型参数冻结/解冻
- 自适应学习率调整
- 参数重要性评估
- 增量更新策略
- 性能监控和回滚
"""

from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from abc import ABC, abstractmethod
import logging
import time
import threading
import json
import os
import copy
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import pickle
import hashlib

from .llm_interface import BaseLLM, LLMConfig, LLMResponse

logger = logging.getLogger(__name__)


class UpdateStrategy(Enum):
    """更新策略"""
    FROZEN = "frozen"  # 完全冻结
    ADAPTIVE = "adaptive"  # 自适应更新
    SELECTIVE = "selective"  # 选择性更新
    INCREMENTAL = "incremental"  # 增量更新
    GRADUAL = "gradual"  # 渐进式更新


class ParameterImportance(Enum):
    """参数重要性级别"""
    CRITICAL = "critical"  # 关键参数
    IMPORTANT = "important"  # 重要参数
    MODERATE = "moderate"  # 中等参数
    LOW = "low"  # 低重要性参数


@dataclass
class FrozenConfig:
    """冻结配置"""
    strategy: UpdateStrategy = UpdateStrategy.ADAPTIVE
    frozen_layers: List[str] = field(default_factory=list)  # 冻结的层
    frozen_parameters: List[str] = field(default_factory=list)  # 冻结的参数
    adaptive_learning_rate: bool = True
    min_learning_rate: float = 1e-6
    max_learning_rate: float = 1e-3
    importance_threshold: float = 0.1
    update_frequency: int = 100  # 更新频率
    performance_window: int = 50  # 性能评估窗口
    rollback_threshold: float = 0.05  # 回滚阈值


@dataclass
class ParameterInfo:
    """参数信息"""
    name: str
    shape: Tuple[int, ...]
    importance: ParameterImportance
    frozen: bool = False
    last_update: float = 0.0
    update_count: int = 0
    gradient_norm: float = 0.0
    sensitivity: float = 0.0


@dataclass
class UpdateHistory:
    """更新历史"""
    timestamp: float
    parameter_name: str
    old_value: Any
    new_value: Any
    learning_rate: float
    performance_change: float
    strategy: UpdateStrategy


class AdaptiveLearningRate:
    """自适应学习率管理器"""
    
    def __init__(self, initial_lr: float = 1e-4, min_lr: float = 1e-6, max_lr: float = 1e-3):
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.current_lr = initial_lr
        self.performance_history = deque(maxlen=100)
        self.lr_history = deque(maxlen=100)
        
    def update(self, performance_metric: float) -> float:
        """根据性能指标更新学习率"""
        self.performance_history.append(performance_metric)
        self.lr_history.append(self.current_lr)
        
        if len(self.performance_history) < 2:
            return self.current_lr
        
        # 计算性能变化趋势
        recent_performance = list(self.performance_history)[-10:]
        if len(recent_performance) >= 2:
            try:
                if NUMPY_AVAILABLE and np is not None and hasattr(np, 'diff') and hasattr(np, 'mean'):
                    performance_trend = np.mean(np.diff(recent_performance))
                else:
                    # 手动计算趋势
                    diffs = [recent_performance[i] - recent_performance[i-1] 
                            for i in range(1, len(recent_performance))]
                    performance_trend = sum(diffs) / len(diffs)
            except:
                performance_trend = 0.0
            
            # 根据趋势调整学习率
            if performance_trend > 0.01:  # 性能提升
                self.current_lr = min(self.max_lr, self.current_lr * 1.1)
            elif performance_trend < -0.01:  # 性能下降
                self.current_lr = max(self.min_lr, self.current_lr * 0.9)
        
        return self.current_lr
    
    def get_lr(self) -> float:
        """获取当前学习率"""
        return self.current_lr
    
    def reset(self) -> None:
        """重置学习率"""
        self.current_lr = self.initial_lr
        self.performance_history.clear()
        self.lr_history.clear()


class ParameterImportanceAnalyzer:
    """参数重要性分析器"""
    
    def __init__(self):
        self.importance_cache = {}
        self.sensitivity_cache = {}
        
    def analyze_importance(self, parameters: Dict[str, Any], 
                          gradients: Dict[str, Any]) -> Dict[str, ParameterImportance]:
        """分析参数重要性"""
        importance_scores = {}
        
        for name, param in parameters.items():
            if name in gradients:
                grad = gradients[name]
                
                # 计算梯度范数
                try:
                    if NUMPY_AVAILABLE and np is not None and isinstance(grad, np.ndarray):
                        grad_norm = np.linalg.norm(grad)
                    elif isinstance(grad, list):
                        grad_norm = sum(g * g for g in grad) ** 0.5
                    elif isinstance(grad, (int, float)):
                        grad_norm = abs(grad)
                    else:
                        grad_norm = abs(grad)
                except (TypeError, ValueError):
                    grad_norm = 0.0
                
                # 计算参数敏感性
                try:
                    if NUMPY_AVAILABLE and np is not None and isinstance(param, np.ndarray):
                        param_norm = np.linalg.norm(param)
                    elif isinstance(param, list):
                        param_norm = sum(p * p for p in param) ** 0.5
                    elif isinstance(param, (int, float)):
                        param_norm = abs(param)
                    else:
                        param_norm = abs(param)
                    
                    sensitivity = grad_norm / (param_norm + 1e-8)
                except (TypeError, ValueError):
                    sensitivity = 0.0
                
                # 根据敏感性和梯度范数确定重要性
                if sensitivity > 0.5 and grad_norm > 0.1:
                    importance = ParameterImportance.CRITICAL
                elif sensitivity > 0.2 and grad_norm > 0.05:
                    importance = ParameterImportance.IMPORTANT
                elif sensitivity > 0.1 and grad_norm > 0.01:
                    importance = ParameterImportance.MODERATE
                else:
                    importance = ParameterImportance.LOW
                
                importance_scores[name] = importance
                self.importance_cache[name] = importance
                self.sensitivity_cache[name] = sensitivity
        
        return importance_scores
    
    def get_cached_importance(self, parameter_name: str) -> Optional[ParameterImportance]:
        """获取缓存的参数重要性"""
        return self.importance_cache.get(parameter_name)
    
    def get_sensitivity(self, parameter_name: str) -> float:
        """获取参数敏感性"""
        return self.sensitivity_cache.get(parameter_name, 0.0)


class FrozenAdaptiveLLM:
    """冻结自适应LLM管理器"""
    
    def __init__(self, base_llm: BaseLLM, config: FrozenConfig):
        self.base_llm = base_llm
        self.config = config
        self.parameter_info = {}
        self.update_history = []
        self.performance_history = deque(maxlen=config.performance_window)
        self.adaptive_lr = AdaptiveLearningRate(
            initial_lr=1e-4,
            min_lr=config.min_learning_rate,
            max_lr=config.max_learning_rate
        )
        self.importance_analyzer = ParameterImportanceAnalyzer()
        self.lock = threading.RLock()
        self.update_counter = 0
        
        # 初始化参数信息
        self._initialize_parameters()
    
    def _initialize_parameters(self) -> None:
        """初始化参数信息"""
        parameters = self.base_llm.get_parameters()
        
        for name, param in parameters.items():
            # 处理不同类型的参数
            if hasattr(param, 'shape'):
                shape = param.shape
            elif isinstance(param, list):
                shape = (len(param),)
            elif isinstance(param, (int, float)):
                shape = (1,)
            else:
                # 对于其他类型，尝试获取长度，如果失败则使用默认形状
                try:
                    shape = (len(param),)
                except (TypeError, AttributeError):
                    shape = (1,)
            
            frozen = name in self.config.frozen_parameters or any(
                layer in name for layer in self.config.frozen_layers
            )
            
            self.parameter_info[name] = ParameterInfo(
                name=name,
                shape=shape,
                importance=ParameterImportance.MODERATE,
                frozen=frozen
            )
    
    def freeze_parameters(self, parameter_names: List[str]) -> None:
        """冻结指定参数"""
        with self.lock:
            for name in parameter_names:
                if name in self.parameter_info:
                    self.parameter_info[name].frozen = True
                    logger.info(f"冻结参数: {name}")
    
    def unfreeze_parameters(self, parameter_names: List[str]) -> None:
        """解冻指定参数"""
        with self.lock:
            for name in parameter_names:
                if name in self.parameter_info:
                    self.parameter_info[name].frozen = False
                    logger.info(f"解冻参数: {name}")
    
    def freeze_layers(self, layer_names: List[str]) -> None:
        """冻结指定层"""
        with self.lock:
            for name in self.parameter_info:
                if any(layer in name for layer in layer_names):
                    self.parameter_info[name].frozen = True
                    logger.info(f"冻结层参数: {name}")
    
    def unfreeze_layers(self, layer_names: List[str]) -> None:
        """解冻指定层"""
        with self.lock:
            for name in self.parameter_info:
                if any(layer in name for layer in layer_names):
                    self.parameter_info[name].frozen = False
                    logger.info(f"解冻层参数: {name}")
    
    def analyze_and_update_importance(self, gradients: Dict[str, Any]) -> Dict[str, ParameterImportance]:
        """分析并更新参数重要性"""
        parameters = self.base_llm.get_parameters()
        importance_scores = self.importance_analyzer.analyze_importance(parameters, gradients)
        
        with self.lock:
            for name, importance in importance_scores.items():
                if name in self.parameter_info:
                    self.parameter_info[name].importance = importance
                    if name in gradients:
                        grad = gradients[name]
                        try:
                            if NUMPY_AVAILABLE and np is not None and isinstance(grad, np.ndarray):
                                grad_norm = np.linalg.norm(grad)
                            elif isinstance(grad, list):
                                grad_norm = sum(g * g for g in grad) ** 0.5
                            elif isinstance(grad, (int, float)):
                                grad_norm = abs(grad)
                            else:
                                grad_norm = abs(grad)
                        except (TypeError, ValueError):
                            grad_norm = 0.0
                        
                        self.parameter_info[name].gradient_norm = grad_norm
                        self.parameter_info[name].sensitivity = self.importance_analyzer.get_sensitivity(name)
        
        return importance_scores
    
    def should_update_parameter(self, parameter_name: str, importance: ParameterImportance) -> bool:
        """判断是否应该更新参数"""
        if self.config.strategy == UpdateStrategy.FROZEN:
            return False
        
        if parameter_name not in self.parameter_info:
            return True
        
        param_info = self.parameter_info[parameter_name]
        
        if param_info.frozen:
            return False
        
        if self.config.strategy == UpdateStrategy.SELECTIVE:
            return importance in [ParameterImportance.CRITICAL, ParameterImportance.IMPORTANT]
        
        if self.config.strategy == UpdateStrategy.INCREMENTAL:
            return self.update_counter % self.config.update_frequency == 0
        
        return True
    
    def update_parameters(self, gradients: Dict[str, Any], 
                         performance_metric: Optional[float] = None) -> Dict[str, Any]:
        """更新参数"""
        with self.lock:
            self.update_counter += 1
            
            # 分析参数重要性
            importance_scores = self.analyze_and_update_importance(gradients)
            
            # 更新学习率
            if self.config.adaptive_learning_rate and performance_metric is not None:
                learning_rate = self.adaptive_lr.update(performance_metric)
            else:
                learning_rate = self.adaptive_lr.get_lr()
            
            # 记录性能
            if performance_metric is not None:
                self.performance_history.append(performance_metric)
            
            # 应用更新策略
            updated_parameters = {}
            parameters = self.base_llm.get_parameters()
            
            for name, grad in gradients.items():
                if name not in parameters:
                    continue
                
                importance = importance_scores.get(name, ParameterImportance.MODERATE)
                
                if not self.should_update_parameter(name, importance):
                    continue
                
                # 计算更新
                if self.config.strategy == UpdateStrategy.GRADUAL:
                    # 渐进式更新
                    update_factor = 1.0 / (1.0 + self.parameter_info[name].update_count * 0.1)
                    learning_rate *= update_factor
                
                old_value = parameters[name]
                new_value = self._apply_gradient(old_value, grad, learning_rate)
                
                # 记录更新历史
                self.update_history.append(UpdateHistory(
                    timestamp=time.time(),
                    parameter_name=name,
                    old_value=copy.deepcopy(old_value),
                    new_value=copy.deepcopy(new_value),
                    learning_rate=learning_rate,
                    performance_change=performance_metric or 0.0,
                    strategy=self.config.strategy
                ))
                
                updated_parameters[name] = new_value
                self.parameter_info[name].update_count += 1
                self.parameter_info[name].last_update = time.time()
            
            # 应用更新到基础LLM
            if updated_parameters:
                self.base_llm.update_parameters(updated_parameters, learning_rate)
            
            return updated_parameters
    
    def _apply_gradient(self, parameter: Any, gradient: Any, learning_rate: float) -> Any:
        """应用梯度更新"""
        try:
            # 检查参数类型，只对数值类型进行更新
            if isinstance(parameter, (dict, str, bool)):
                # 对于非数值类型，直接返回原值
                logger.debug(f"跳过非数值类型参数更新: {type(parameter)}")
                return parameter
            
            if NUMPY_AVAILABLE and np is not None and isinstance(parameter, np.ndarray) and isinstance(gradient, np.ndarray):
                return parameter - learning_rate * gradient
            elif isinstance(parameter, list) and isinstance(gradient, list):
                return [p - learning_rate * g for p, g in zip(parameter, gradient)]
            elif isinstance(parameter, (int, float)) and isinstance(gradient, (int, float)):
                return parameter - learning_rate * gradient
            else:
                # 对于其他类型，尝试进行数值运算
                return parameter - learning_rate * gradient
        except (TypeError, ValueError):
            # 如果无法进行数值运算，返回原参数
            logger.warning(f"无法更新参数，保持原值: {type(parameter)}")
            return parameter
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """生成响应"""
        return self.base_llm.generate(prompt, **kwargs)
    
    def get_parameters(self) -> Dict[str, Any]:
        """获取参数"""
        return self.base_llm.get_parameters()
    
    def get_parameter_info(self) -> Dict[str, ParameterInfo]:
        """获取参数信息"""
        with self.lock:
            return copy.deepcopy(self.parameter_info)
    
    def get_update_history(self) -> List[UpdateHistory]:
        """获取更新历史"""
        with self.lock:
            return copy.deepcopy(self.update_history)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        with self.lock:
            if not self.performance_history:
                return {}
            
            performance_list = list(self.performance_history)
            
            # 计算基本统计信息
            current_performance = performance_list[-1]
            average_performance = sum(performance_list) / len(performance_list)
            
            # 计算性能趋势
            if len(performance_list) > 1:
                recent_performance = performance_list[-10:] if len(performance_list) >= 10 else performance_list
                performance_trend = sum(recent_performance[i] - recent_performance[i-1] 
                                      for i in range(1, len(recent_performance))) / (len(recent_performance) - 1)
            else:
                performance_trend = 0.0
            
            # 计算标准差
            try:
                if NUMPY_AVAILABLE and np is not None and hasattr(np, 'std'):
                    performance_std = np.std(performance_list)
                else:
                    # 手动计算标准差
                    mean = average_performance
                    variance = sum((x - mean) ** 2 for x in performance_list) / len(performance_list)
                    performance_std = variance ** 0.5
            except:
                performance_std = 0.0
            
            return {
                "current_performance": current_performance,
                "average_performance": average_performance,
                "performance_trend": performance_trend,
                "performance_std": performance_std,
                "update_count": self.update_counter,
                "current_learning_rate": self.adaptive_lr.get_lr()
            }
    
    def rollback_to_checkpoint(self, checkpoint_path: str) -> bool:
        """回滚到检查点"""
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
            
            # 恢复参数
            self.base_llm.update_parameters(checkpoint['parameters'], 0.0)
            
            # 恢复状态
            self.parameter_info = checkpoint['parameter_info']
            self.update_history = checkpoint['update_history']
            self.performance_history = deque(checkpoint['performance_history'], 
                                           maxlen=self.config.performance_window)
            self.adaptive_lr.current_lr = checkpoint['learning_rate']
            self.update_counter = checkpoint['update_counter']
            
            logger.info(f"成功回滚到检查点: {checkpoint_path}")
            return True
            
        except Exception as e:
            logger.error(f"回滚失败: {e}")
            return False
    
    def save_checkpoint(self, checkpoint_path: str) -> bool:
        """保存检查点"""
        try:
            checkpoint = {
                'parameters': self.base_llm.get_parameters(),
                'parameter_info': self.parameter_info,
                'update_history': self.update_history,
                'performance_history': list(self.performance_history),
                'learning_rate': self.adaptive_lr.get_lr(),
                'update_counter': self.update_counter,
                'config': self.config
            }
            
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint, f)
            
            logger.info(f"检查点已保存: {checkpoint_path}")
            return True
            
        except Exception as e:
            logger.error(f"保存检查点失败: {e}")
            return False
    
    def export_config(self, config_path: str) -> bool:
        """导出配置"""
        try:
            config_data = {
                'frozen_config': {
                    'strategy': self.config.strategy.value,
                    'frozen_layers': self.config.frozen_layers,
                    'frozen_parameters': self.config.frozen_parameters,
                    'adaptive_learning_rate': self.config.adaptive_learning_rate,
                    'min_learning_rate': self.config.min_learning_rate,
                    'max_learning_rate': self.config.max_learning_rate,
                    'importance_threshold': self.config.importance_threshold,
                    'update_frequency': self.config.update_frequency,
                    'performance_window': self.config.performance_window,
                    'rollback_threshold': self.config.rollback_threshold
                },
                'parameter_info': {
                    name: {
                        'importance': info.importance.value,
                        'frozen': info.frozen,
                        'update_count': info.update_count,
                        'sensitivity': info.sensitivity
                    }
                    for name, info in self.parameter_info.items()
                }
            }
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"配置已导出: {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"导出配置失败: {e}")
            return False


class FrozenAdaptiveManager:
    """冻结自适应管理器"""
    
    def __init__(self):
        self.models = {}
        self.configs = {}
        self.lock = threading.RLock()
    
    def register_model(self, model_id: str, base_llm: BaseLLM, 
                      config: FrozenConfig) -> FrozenAdaptiveLLM:
        """注册模型"""
        with self.lock:
            if model_id in self.models:
                logger.warning(f"模型 {model_id} 已存在，将被覆盖")
            
            frozen_llm = FrozenAdaptiveLLM(base_llm, config)
            self.models[model_id] = frozen_llm
            self.configs[model_id] = config
            
            logger.info(f"注册模型: {model_id}")
            return frozen_llm
    
    def get_model(self, model_id: str) -> Optional[FrozenAdaptiveLLM]:
        """获取模型"""
        return self.models.get(model_id)
    
    def remove_model(self, model_id: str) -> bool:
        """移除模型"""
        with self.lock:
            if model_id in self.models:
                del self.models[model_id]
                del self.configs[model_id]
                logger.info(f"移除模型: {model_id}")
                return True
            return False
    
    def list_models(self) -> List[str]:
        """列出所有模型"""
        return list(self.models.keys())
    
    def get_model_stats(self, model_id: str) -> Dict[str, Any]:
        """获取模型统计信息"""
        model = self.get_model(model_id)
        if not model:
            return {}
        
        return {
            'model_id': model_id,
            'config': self.configs[model_id],
            'parameter_info': model.get_parameter_info(),
            'performance_stats': model.get_performance_stats(),
            'update_history_count': len(model.get_update_history())
        }


# 工厂函数
def create_frozen_config(
    strategy: Union[str, UpdateStrategy] = "adaptive",
    frozen_layers: Optional[List[str]] = None,
    frozen_parameters: Optional[List[str]] = None,
    **kwargs
) -> FrozenConfig:
    """创建冻结配置"""
    if isinstance(strategy, str):
        strategy = UpdateStrategy(strategy)
    
    return FrozenConfig(
        strategy=strategy,
        frozen_layers=frozen_layers or [],
        frozen_parameters=frozen_parameters or [],
        **kwargs
    )


def create_frozen_adaptive_llm(
    base_llm: BaseLLM,
    strategy: Union[str, UpdateStrategy] = "adaptive",
    **kwargs
) -> FrozenAdaptiveLLM:
    """创建冻结自适应LLM"""
    config = create_frozen_config(strategy, **kwargs)
    return FrozenAdaptiveLLM(base_llm, config)


def create_frozen_adaptive_manager() -> FrozenAdaptiveManager:
    """创建冻结自适应管理器"""
    return FrozenAdaptiveManager()


# 预设配置
def get_preset_configs() -> Dict[str, FrozenConfig]:
    """获取预设配置"""
    return {
        "conservative": FrozenConfig(
            strategy=UpdateStrategy.SELECTIVE,
            frozen_layers=["embedding", "encoder"],
            adaptive_learning_rate=True,
            min_learning_rate=1e-6,
            max_learning_rate=1e-4,
            importance_threshold=0.2
        ),
        "aggressive": FrozenConfig(
            strategy=UpdateStrategy.ADAPTIVE,
            frozen_layers=[],
            adaptive_learning_rate=True,
            min_learning_rate=1e-5,
            max_learning_rate=1e-3,
            importance_threshold=0.05
        ),
        "balanced": FrozenConfig(
            strategy=UpdateStrategy.GRADUAL,
            frozen_layers=["embedding"],
            adaptive_learning_rate=True,
            min_learning_rate=1e-6,
            max_learning_rate=1e-3,
            importance_threshold=0.1
        ),
        "frozen": FrozenConfig(
            strategy=UpdateStrategy.FROZEN,
            frozen_layers=["embedding", "encoder", "decoder"],
            adaptive_learning_rate=False
        )
    } 