#!/usr/bin/env python3
"""
简化版多模型单环境训练系统
支持协同、竞争、组队博弈等不同训练模式
使用VLLM/AReaL中集成的不同LoRA实现
"""

import asyncio
import time
import logging
import random
import json
import os
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor

# VLLM集成 - 使用Camel和Oasis接口
try:
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType
    import oasis
    CAMEL_OASIS_AVAILABLE = True
    print("✅ Camel和Oasis VLLM接口可用")
except ImportError:
    CAMEL_OASIS_AVAILABLE = False
    print("⚠️ Camel和Oasis VLLM接口不可用，将使用模拟模式")

# 备用HTTP客户端
try:
    import requests
    import aiohttp
    VLLM_AVAILABLE = True
    print("✅ VLLM HTTP客户端可用")
except ImportError:
    VLLM_AVAILABLE = False
    print("⚠️ VLLM HTTP客户端不可用，将使用模拟模式")

# 简化的numpy替代
def simple_randn(*shape):
    """简化的随机数生成函数"""
    if len(shape) == 1:
        return [random.uniform(-1, 1) for _ in range(shape[0])]
    elif len(shape) == 2:
        return [[random.uniform(-1, 1) for _ in range(shape[1])] 
               for _ in range(shape[0])]
    else:
        return [random.uniform(-1, 1) for _ in range(shape[0])]

class SimpleNumpy:
    @staticmethod
    def mean(data):
        if not data:
            return 0.0
        return sum(data) / len(data)
    
    @staticmethod
    def array(data):
        return data
    
    @staticmethod
    def zeros(shape):
        if len(shape) == 1:
            return [0.0] * shape[0]
        elif len(shape) == 2:
            return [[0.0] * shape[1] for _ in range(shape[0])]
        return [0.0] * shape[0]

np = SimpleNumpy()

logger = logging.getLogger(__name__)

class VLLMClient:
    """VLLM客户端 - 使用Camel和Oasis接口"""
    
    def __init__(self, url: str = "http://localhost:8001/v1", model_name: str = "qwen-2"):
        self.url = url
        self.model_name = model_name
        self.camel_model = None
        self.connection_available = False
        self._initialize_camel_model()
    
    def _initialize_camel_model(self):
        """初始化Camel模型"""
        if CAMEL_OASIS_AVAILABLE:
            try:
                self.camel_model = ModelFactory.create(
                    model_platform=ModelPlatformType.VLLM,
                    model_type=self.model_name,
                    url=self.url,
                )
                self.connection_available = True
                print(f"✅ Camel VLLM模型初始化成功: {self.url}")
            except Exception as e:
                print(f"⚠️ Camel VLLM模型初始化失败: {e}")
                self.connection_available = False
        else:
            print("⚠️ Camel和Oasis不可用，将使用模拟模式")
            self.connection_available = False
    
    async def generate(self, prompt: str, max_tokens: int = 100) -> str:
        """生成文本响应"""
        if self.camel_model and CAMEL_OASIS_AVAILABLE and self.connection_available:
            try:
                # 使用Camel模型生成响应
                response = await self.camel_model.generate(prompt, max_tokens=max_tokens)
                print(f"🤖 Camel VLLM生成: {response[:50]}...")
                return response
            except Exception as e:
                print(f"❌ Camel VLLM调用失败: {e}")
                print("回退到模拟模式")
        
        # 备用HTTP客户端
        if VLLM_AVAILABLE:
            try:
                async with aiohttp.ClientSession() as session:
                    payload = {
                        "model": self.model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": 0.7
                    }
                    
                    async with session.post(f"{self.url}/chat/completions", json=payload) as response:
                        if response.status == 200:
                            result = await response.json()
                            content = result["choices"][0]["message"]["content"]
                            print(f"🤖 HTTP VLLM生成: {content[:50]}...")
                            return content
                        else:
                            print(f"❌ HTTP VLLM响应错误: {response.status}")
                            return self._generate_mock_response(prompt)
            except Exception as e:
                print(f"❌ HTTP VLLM调用失败: {e}")
                return self._generate_mock_response(prompt)
        
        return self._generate_mock_response(prompt)
    
    def _generate_mock_response(self, prompt: str) -> str:
        """生成模拟响应"""
        mock_responses = [
            "基于当前任务分析，我建议采用协同策略。",
            "通过竞争机制可以获得更好的性能表现。",
            "团队合作是解决复杂问题的关键。",
            "需要平衡效率和准确性的关系。",
            "LoRA适应可以帮助模型更好地完成任务。"
        ]
        return random.choice(mock_responses)

class TrainingMode(Enum):
    """训练模式"""
    COOPERATIVE = "cooperative"      # 协同训练
    COMPETITIVE = "competitive"      # 竞争训练
    TEAM_BATTLE = "team_battle"      # 组队博弈
    MIXED = "mixed"                  # 混合模式

class ModelRole(Enum):
    """模型角色"""
    LEADER = "leader"                # 领导者
    FOLLOWER = "follower"            # 跟随者
    COMPETITOR = "competitor"        # 竞争者
    TEAMMATE = "teammate"            # 队友
    NEUTRAL = "neutral"              # 中立者

@dataclass
class ModelConfig:
    """模型配置"""
    model_id: str
    model_name: str
    role: ModelRole
    lora_rank: int = 8
    lora_alpha: float = 16.0
    learning_rate: float = 1e-4
    team_id: Optional[str] = None
    specialization: str = "general"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "role": self.role.value,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "learning_rate": self.learning_rate,
            "team_id": self.team_id,
            "specialization": self.specialization
        }

@dataclass
class TrainingTask:
    """训练任务"""
    task_id: str
    task_type: str
    difficulty: float  # 0-1
    reward_pool: float
    max_steps: int
    required_models: int
    cooperation_level: float  # 0-1, 0=完全竞争, 1=完全协同
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "difficulty": self.difficulty,
            "reward_pool": self.reward_pool,
            "max_steps": self.max_steps,
            "required_models": self.required_models,
            "cooperation_level": self.cooperation_level,
            "created_at": self.created_at.isoformat()
        }

@dataclass
class ModelPerformance:
    """模型性能指标"""
    model_id: str
    task_id: str
    completion_time: float
    accuracy: float
    efficiency: float
    cooperation_score: float
    reward_earned: float
    weight_updates: int
    lora_adaptations: int
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "task_id": self.task_id,
            "completion_time": self.completion_time,
            "accuracy": self.accuracy,
            "efficiency": self.efficiency,
            "cooperation_score": self.cooperation_score,
            "reward_earned": self.reward_earned,
            "weight_updates": self.weight_updates,
            "lora_adaptations": self.lora_adaptations,
            "timestamp": self.timestamp.isoformat()
        }

class MockLoRAManager:
    """模拟LoRA管理器"""
    
    def __init__(self, rank: int = 8, alpha: float = 16.0):
        self.rank = rank
        self.alpha = alpha
        self.adaptation_count = 0
        self.adaptations = []
    
    def adapt(self, task_type: str, performance: float):
        """模拟LoRA适应"""
        self.adaptation_count += 1
        self.adaptations.append({
            "task_type": task_type,
            "performance": performance,
            "timestamp": datetime.now().isoformat()
        })
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "adaptation_count": self.adaptation_count,
            "recent_adaptations": self.adaptations[-5:] if self.adaptations else []
        }

class MockFrozenLLM:
    """模拟Frozen Adaptive LLM"""
    
    def __init__(self, learning_rate: float = 1e-4):
        self.learning_rate = learning_rate
        self.weight_update_count = 0
        self.parameters = {
            "embedding": simple_randn(100, 768),
            "layers.0": simple_randn(768, 768),
            "layers.1": simple_randn(768, 768),
            "output": simple_randn(768, 100)
        }
    
    def update_parameters(self, gradients: Dict[str, Any], lr: float):
        """更新参数"""
        self.weight_update_count += 1
        # 模拟参数更新
        for key in self.parameters:
            if key in gradients:
                # 简化的参数更新
                pass
        return True
    
    def generate(self, prompt: str) -> str:
        """生成响应"""
        return f"Mock response for {prompt[:20]}..."
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "weight_update_count": self.weight_update_count,
            "parameter_count": len(self.parameters)
        }

class LoRAModel:
    """LoRA模型封装"""
    
    def __init__(self, config: ModelConfig, vllm_url: str = "http://localhost:8001/v1"):
        self.config = config
        self.vllm_url = vllm_url
        self.lora_manager = MockLoRAManager(config.lora_rank, config.lora_alpha)
        self.frozen_llm = MockFrozenLLM(config.learning_rate)
        self.performance_history: List[ModelPerformance] = []
        self.current_task = None
        
        # 初始化VLLM客户端
        self.vllm_client = VLLMClient(vllm_url, f"{config.model_name}-lora-{config.lora_rank}")
        
        print(f"✅ Model {self.config.model_id} initialized with LoRA rank={self.config.lora_rank}")
        print(f"🤖 VLLM客户端: {self.vllm_client.connection_available}")
    
    async def process_task(self, task: TrainingTask, other_models: List['LoRAModel']) -> ModelPerformance:
        """处理训练任务"""
        start_time = time.time()
        self.current_task = task
        
        # 根据任务类型和合作级别决定策略
        strategy = self._determine_strategy(task, other_models)
        
        # 执行任务
        result = await self._execute_task(task, strategy, other_models)
        
        # 计算性能指标
        completion_time = time.time() - start_time
        performance = ModelPerformance(
            model_id=self.config.model_id,
            task_id=task.task_id,
            completion_time=completion_time,
            accuracy=result.get("accuracy", 0.0),
            efficiency=result.get("efficiency", 0.0),
            cooperation_score=result.get("cooperation_score", 0.0),
            reward_earned=result.get("reward_earned", 0.0),
            weight_updates=self.frozen_llm.weight_update_count,
            lora_adaptations=self.lora_manager.adaptation_count
        )
        
        self.performance_history.append(performance)
        
        # 更新模型权重
        await self._update_weights(result)
        
        return performance
    
    def _determine_strategy(self, task: TrainingTask, other_models: List['LoRAModel']) -> Dict[str, Any]:
        """根据任务和合作级别确定策略"""
        strategy = {
            "mode": "cooperative" if task.cooperation_level > 0.5 else "competitive",
            "teamwork_level": task.cooperation_level,
            "communication": task.cooperation_level > 0.3,
            "resource_sharing": task.cooperation_level > 0.7
        }
        
        # 根据角色调整策略
        if self.config.role == ModelRole.LEADER:
            strategy["leadership"] = True
            strategy["decision_making"] = "centralized"
        elif self.config.role == ModelRole.FOLLOWER:
            strategy["leadership"] = False
            strategy["decision_making"] = "delegated"
        elif self.config.role == ModelRole.COMPETITOR:
            strategy["mode"] = "competitive"
            strategy["aggression"] = 0.8
        
        return strategy
    
    async def _execute_task(self, task: TrainingTask, strategy: Dict[str, Any], 
                          other_models: List['LoRAModel']) -> Dict[str, Any]:
        """执行任务"""
        # 构建VLLM提示词
        prompt = self._build_task_prompt(task, strategy, other_models)
        
        # 使用VLLM生成响应
        try:
            vllm_response = await self.vllm_client.generate(prompt, max_tokens=150)
            print(f"🤖 Model {self.config.model_id} VLLM响应: {vllm_response[:100]}...")
        except Exception as e:
            print(f"❌ VLLM调用失败: {e}")
            vllm_response = "使用默认策略执行任务"
        
        # 根据VLLM响应和策略计算性能
        base_accuracy = random.uniform(0.6, 0.9)
        base_efficiency = random.uniform(0.5, 0.8)
        
        # VLLM响应质量加成
        vllm_bonus = 0.1 if "协同" in vllm_response or "合作" in vllm_response else 0.05
        if "竞争" in vllm_response or "优化" in vllm_response:
            vllm_bonus += 0.05
        
        # 合作加成
        cooperation_bonus = strategy["teamwork_level"] * 0.2
        accuracy = min(1.0, base_accuracy + cooperation_bonus + vllm_bonus)
        efficiency = min(1.0, base_efficiency + cooperation_bonus + vllm_bonus)
        
        # 计算奖励
        reward_earned = task.reward_pool * accuracy * efficiency / len(other_models + [self])
        
        # 模拟LoRA适应
        if random.random() < 0.3:
            self.lora_manager.adapt(task.task_type, accuracy)
        
        return {
            "accuracy": accuracy,
            "efficiency": efficiency,
            "cooperation_score": strategy["teamwork_level"],
            "reward_earned": reward_earned,
            "vllm_response": vllm_response,
            "strategy_used": strategy
        }
    
    def _build_task_prompt(self, task: TrainingTask, strategy: Dict[str, Any], 
                          other_models: List['LoRAModel']) -> str:
        """构建VLLM任务提示词"""
        prompt = f"""
你是一个AI模型，正在参与多模型训练任务。

任务信息:
- 任务ID: {task.task_id}
- 任务类型: {task.task_type}
- 难度: {task.difficulty:.2f}
- 奖励池: {task.reward_pool:.2f}
- 合作级别: {task.cooperation_level:.2f}

你的角色: {self.config.role.value}
你的团队: {self.config.team_id or '无'}
你的专长: {self.config.specialization}

当前策略:
- 模式: {strategy.get('mode', 'unknown')}
- 团队合作级别: {strategy.get('teamwork_level', 0):.2f}
- 通信: {strategy.get('communication', False)}
- 资源共享: {strategy.get('resource_sharing', False)}

其他模型数量: {len(other_models)}

请根据以上信息，为这个任务提供执行策略建议。考虑你的角色、任务类型和合作级别。
"""
        return prompt
    
    async def _update_weights(self, result: Dict[str, Any]):
        """更新模型权重"""
        if result.get("accuracy", 0) > 0.7:
            # 生成模拟梯度
            gradients = {
                "embedding": simple_randn(100, 768),
                "layers.0": simple_randn(768, 768),
                "layers.1": simple_randn(768, 768),
                "output": simple_randn(768, 100)
            }
            
            try:
                self.frozen_llm.update_parameters(gradients, self.config.learning_rate)
                print(f"🔄 Model {self.config.model_id} weights updated")
            except Exception as e:
                print(f"⚠️ Failed to update weights for model {self.config.model_id}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取模型统计信息"""
        if not self.performance_history:
            return {"model_id": self.config.model_id, "no_data": True}
        
        recent_performances = self.performance_history[-10:]  # 最近10次
        
        return {
            "model_id": self.config.model_id,
            "role": self.config.role.value,
            "team_id": self.config.team_id,
            "total_tasks": len(self.performance_history),
            "avg_accuracy": np.mean([p.accuracy for p in recent_performances]),
            "avg_efficiency": np.mean([p.efficiency for p in recent_performances]),
            "avg_cooperation": np.mean([p.cooperation_score for p in recent_performances]),
            "total_reward": sum([p.reward_earned for p in self.performance_history]),
            "weight_updates": self.frozen_llm.weight_update_count,
            "lora_adaptations": self.lora_manager.adaptation_count,
            "last_activity": self.performance_history[-1].timestamp.isoformat() if self.performance_history else None,
            "lora_stats": self.lora_manager.get_stats(),
            "llm_stats": self.frozen_llm.get_stats()
        }

class MultiModelEnvironment:
    """多模型单环境训练系统"""
    
    def __init__(self, 
                 vllm_url: str = "http://localhost:8001/v1",
                 training_mode: TrainingMode = TrainingMode.MIXED,
                 max_models: int = 10):
        self.vllm_url = vllm_url
        self.training_mode = training_mode
        self.max_models = max_models
        
        self.models: Dict[str, LoRAModel] = {}
        self.tasks: List[TrainingTask] = []
        self.task_queue: List[TrainingTask] = []
        self.completed_tasks: List[TrainingTask] = []
        
        # 测试VLLM连接
        self.vllm_available = self._test_vllm_connection()
        
        self._generate_initial_tasks()
        print(f"✅ Multi-model environment initialized with mode: {training_mode.value}")
        print(f"🤖 VLLM可用性: {self.vllm_available}")
    
    def _test_vllm_connection(self) -> bool:
        """测试VLLM连接"""
        try:
            response = requests.get(f"{self.vllm_url}/models", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def _generate_initial_tasks(self):
        """生成初始训练任务"""
        task_types = ["classification", "generation", "reasoning", "optimization", "collaboration"]
        
        for i in range(20):
            task = TrainingTask(
                task_id=f"task_{i:03d}",
                task_type=random.choice(task_types),
                difficulty=random.uniform(0.3, 0.9),
                reward_pool=random.uniform(10.0, 100.0),
                max_steps=random.randint(5, 20),
                required_models=random.randint(2, 5),
                cooperation_level=random.uniform(0.0, 1.0)
            )
            self.task_queue.append(task)
    
    def add_model(self, config: ModelConfig) -> bool:
        """添加模型到环境"""
        if len(self.models) >= self.max_models:
            print(f"❌ Maximum models ({self.max_models}) reached")
            return False
        
        if config.model_id in self.models:
            print(f"❌ Model {config.model_id} already exists")
            return False
        
        model = LoRAModel(config, self.vllm_url)
        self.models[config.model_id] = model
        
        print(f"✅ Model {config.model_id} added to environment")
        return True
    
    def remove_model(self, model_id: str) -> bool:
        """从环境中移除模型"""
        if model_id not in self.models:
            return False
        
        del self.models[model_id]
        print(f"✅ Model {model_id} removed from environment")
        return True
    
    async def run_training_cycle(self, cycles: int = 5) -> List[Dict[str, Any]]:
        """运行训练周期"""
        results = []
        
        for cycle in range(cycles):
            print(f"\n🔄 Training Cycle {cycle + 1}/{cycles}")
            
            # 分配任务给模型
            cycle_results = await self._execute_training_cycle()
            results.extend(cycle_results)
            
            # 短暂休息
            await asyncio.sleep(1.0)
        
        return results
    
    async def _execute_training_cycle(self) -> List[Dict[str, Any]]:
        """执行单个训练周期"""
        results = []
        
        # 从队列中获取任务
        if not self.task_queue:
            self._generate_initial_tasks()
        
        # 选择适合的任务
        available_models = list(self.models.values())
        if len(available_models) < 2:
            print("⚠️ Need at least 2 models for training")
            return results
        
        # 随机选择任务
        task = random.choice(self.task_queue)
        self.task_queue.remove(task)
        
        print(f"📋 Executing task: {task.task_id} ({task.task_type})")
        print(f"   Cooperation level: {task.cooperation_level:.2f}")
        print(f"   Required models: {task.required_models}")
        print(f"   VLLM available: {self.vllm_available}")
        
        # 选择参与模型
        participating_models = random.sample(available_models, 
                                           min(task.required_models, len(available_models)))
        
        # 并行执行任务
        tasks = []
        for model in participating_models:
            other_models = [m for m in participating_models if m != model]
            task_coro = model.process_task(task, other_models)
            tasks.append(task_coro)
        
        performances = await asyncio.gather(*tasks)
        
        # 记录结果
        for performance in performances:
            results.append(performance.to_dict())
            print(f"   Model {performance.model_id}: accuracy={performance.accuracy:.3f}, "
                  f"efficiency={performance.efficiency:.3f}, reward={performance.reward_earned:.2f}")
        
        self.completed_tasks.append(task)
        return results
    
    def get_environment_stats(self) -> Dict[str, Any]:
        """获取环境统计信息"""
        model_stats = {model_id: model.get_stats() for model_id, model in self.models.items()}
        
        return {
            "environment_info": {
                "training_mode": self.training_mode.value,
                "total_models": len(self.models),
                "max_models": self.max_models,
                "total_tasks": len(self.completed_tasks),
                "pending_tasks": len(self.task_queue)
            },
            "model_stats": model_stats,
            "task_distribution": {
                "completed": len(self.completed_tasks),
                "pending": len(self.task_queue),
                "total_generated": len(self.completed_tasks) + len(self.task_queue)
            },
            "performance_summary": {
                "avg_accuracy": np.mean([stats.get("avg_accuracy", 0.0) for stats in model_stats.values()]),
                "avg_efficiency": np.mean([stats.get("avg_efficiency", 0.0) for stats in model_stats.values()]),
                "total_reward": sum([stats.get("total_reward", 0.0) for stats in model_stats.values()]),
                "total_weight_updates": sum([stats.get("weight_updates", 0) for stats in model_stats.values()]),
                "total_lora_adaptations": sum([stats.get("lora_adaptations", 0) for stats in model_stats.values()])
            }
        }

def create_cooperative_team() -> List[ModelConfig]:
    """创建协同团队"""
    return [
        ModelConfig(
            model_id="leader_001",
            model_name="qwen-2-leader",
            role=ModelRole.LEADER,
            lora_rank=16,
            team_id="team_alpha",
            specialization="strategy"
        ),
        ModelConfig(
            model_id="follower_001",
            model_name="qwen-2-follower",
            role=ModelRole.FOLLOWER,
            lora_rank=8,
            team_id="team_alpha",
            specialization="execution"
        ),
        ModelConfig(
            model_id="follower_002",
            model_name="qwen-2-follower",
            role=ModelRole.FOLLOWER,
            lora_rank=8,
            team_id="team_alpha",
            specialization="analysis"
        )
    ]

def create_competitive_models() -> List[ModelConfig]:
    """创建竞争模型"""
    return [
        ModelConfig(
            model_id="competitor_001",
            model_name="qwen-2-competitor",
            role=ModelRole.COMPETITOR,
            lora_rank=12,
            specialization="aggressive"
        ),
        ModelConfig(
            model_id="competitor_002",
            model_name="qwen-2-competitor",
            role=ModelRole.COMPETITOR,
            lora_rank=12,
            specialization="defensive"
        ),
        ModelConfig(
            model_id="neutral_001",
            model_name="qwen-2-neutral",
            role=ModelRole.NEUTRAL,
            lora_rank=8,
            specialization="balanced"
        )
    ]

def create_team_battle_models() -> List[ModelConfig]:
    """创建组队博弈模型"""
    return [
        # Team Alpha
        ModelConfig(
            model_id="alpha_leader",
            model_name="qwen-2-alpha",
            role=ModelRole.LEADER,
            lora_rank=16,
            team_id="team_alpha",
            specialization="offensive"
        ),
        ModelConfig(
            model_id="alpha_support",
            model_name="qwen-2-alpha",
            role=ModelRole.TEAMMATE,
            lora_rank=8,
            team_id="team_alpha",
            specialization="support"
        ),
        # Team Beta
        ModelConfig(
            model_id="beta_leader",
            model_name="qwen-2-beta",
            role=ModelRole.LEADER,
            lora_rank=16,
            team_id="team_beta",
            specialization="defensive"
        ),
        ModelConfig(
            model_id="beta_support",
            model_name="qwen-2-beta",
            role=ModelRole.TEAMMATE,
            lora_rank=8,
            team_id="team_beta",
            specialization="support"
        )
    ]

async def demo_cooperative_training():
    """演示协同训练"""
    print("\n🤝 Cooperative Training Demo")
    print("=" * 50)
    
    env = MultiModelEnvironment(
        training_mode=TrainingMode.COOPERATIVE,
        max_models=5
    )
    
    # 添加协同团队
    team_configs = create_cooperative_team()
    for config in team_configs:
        env.add_model(config)
    
    # 运行训练
    results = await env.run_training_cycle(cycles=3)
    
    # 显示结果
    stats = env.get_environment_stats()
    print(f"\n📊 Cooperative Training Results:")
    print(f"   Total models: {stats['environment_info']['total_models']}")
    print(f"   Completed tasks: {stats['environment_info']['total_tasks']}")
    print(f"   Average accuracy: {stats['performance_summary']['avg_accuracy']:.3f}")
    print(f"   Average efficiency: {stats['performance_summary']['avg_efficiency']:.3f}")
    print(f"   Total reward: {stats['performance_summary']['total_reward']:.2f}")
    
    return results

async def demo_competitive_training():
    """演示竞争训练"""
    print("\n⚔️ Competitive Training Demo")
    print("=" * 50)
    
    env = MultiModelEnvironment(
        training_mode=TrainingMode.COMPETITIVE,
        max_models=5
    )
    
    # 添加竞争模型
    competitor_configs = create_competitive_models()
    for config in competitor_configs:
        env.add_model(config)
    
    # 运行训练
    results = await env.run_training_cycle(cycles=3)
    
    # 显示结果
    stats = env.get_environment_stats()
    print(f"\n📊 Competitive Training Results:")
    print(f"   Total models: {stats['environment_info']['total_models']}")
    print(f"   Completed tasks: {stats['environment_info']['total_tasks']}")
    print(f"   Average accuracy: {stats['performance_summary']['avg_accuracy']:.3f}")
    print(f"   Average efficiency: {stats['performance_summary']['avg_efficiency']:.3f}")
    print(f"   Total reward: {stats['performance_summary']['total_reward']:.2f}")
    
    return results

async def demo_team_battle():
    """演示组队博弈"""
    print("\n🏆 Team Battle Demo")
    print("=" * 50)
    
    env = MultiModelEnvironment(
        training_mode=TrainingMode.TEAM_BATTLE,
        max_models=6
    )
    
    # 添加组队模型
    team_configs = create_team_battle_models()
    for config in team_configs:
        env.add_model(config)
    
    # 运行训练
    results = await env.run_training_cycle(cycles=4)
    
    # 显示结果
    stats = env.get_environment_stats()
    print(f"\n📊 Team Battle Results:")
    print(f"   Total models: {stats['environment_info']['total_models']}")
    print(f"   Completed tasks: {stats['environment_info']['total_tasks']}")
    print(f"   Average accuracy: {stats['performance_summary']['avg_accuracy']:.3f}")
    print(f"   Average efficiency: {stats['performance_summary']['avg_efficiency']:.3f}")
    print(f"   Total reward: {stats['performance_summary']['total_reward']:.2f}")
    
    # 显示团队表现
    team_performance = {}
    for model_id, model_stats in stats['model_stats'].items():
        team_id = model_stats.get('team_id', 'no_team')
        if team_id not in team_performance:
            team_performance[team_id] = []
        team_performance[team_id].append(model_stats.get('total_reward', 0.0))
    
    print(f"\n🏅 Team Performance:")
    for team_id, rewards in team_performance.items():
        total_reward = sum(rewards)
        avg_reward = total_reward / len(rewards)
        print(f"   {team_id}: total={total_reward:.2f}, avg={avg_reward:.2f}")
    
    return results

async def demo_mixed_training():
    """演示混合训练模式"""
    print("\n🎭 Mixed Training Mode Demo")
    print("=" * 50)
    
    env = MultiModelEnvironment(
        training_mode=TrainingMode.MIXED,
        max_models=8
    )
    
    # 添加各种类型的模型
    all_configs = (
        create_cooperative_team() + 
        create_competitive_models() + 
        create_team_battle_models()
    )
    
    for config in all_configs[:8]:  # 限制数量
        env.add_model(config)
    
    # 运行训练
    results = await env.run_training_cycle(cycles=5)
    
    # 显示结果
    stats = env.get_environment_stats()
    print(f"\n📊 Mixed Training Results:")
    print(f"   Total models: {stats['environment_info']['total_models']}")
    print(f"   Completed tasks: {stats['environment_info']['total_tasks']}")
    print(f"   Average accuracy: {stats['performance_summary']['avg_accuracy']:.3f}")
    print(f"   Average efficiency: {stats['performance_summary']['avg_efficiency']:.3f}")
    print(f"   Total reward: {stats['performance_summary']['total_reward']:.2f}")
    print(f"   Total weight updates: {stats['performance_summary']['total_weight_updates']}")
    print(f"   Total LoRA adaptations: {stats['performance_summary']['total_lora_adaptations']}")
    
    return results

async def main():
    """主函数"""
    print("🚀 Multi-Model Single Environment Training System (Simplified)")
    print("=" * 70)
    print("🤖 VLLM Integration: Enabled")
    print(f"📡 VLLM HTTP Client: {'Available' if VLLM_AVAILABLE else 'Not Available'}")
    print(f"🔗 Camel & Oasis VLLM Client: {'Available' if CAMEL_OASIS_AVAILABLE else 'Not Available'}")
    print("=" * 70)
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    try:
        # 演示不同的训练模式
        results = []
        
        # 1. 协同训练
        cooperative_results = await demo_cooperative_training()
        results.extend(cooperative_results)
        
        # 2. 竞争训练
        competitive_results = await demo_competitive_training()
        results.extend(competitive_results)
        
        # 3. 组队博弈
        team_battle_results = await demo_team_battle()
        results.extend(team_battle_results)
        
        # 4. 混合训练
        mixed_results = await demo_mixed_training()
        results.extend(mixed_results)
        
        # 保存结果
        with open("multi_model_training_simple_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n🎉 Training completed! Total results: {len(results)}")
        print("📁 Results saved to: multi_model_training_simple_results.json")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
