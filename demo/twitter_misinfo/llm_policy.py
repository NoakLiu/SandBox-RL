#!/usr/bin/env python3
"""
Twitter Misinformation LLM 策略
集成 SandGraph Core 组件，支持更复杂的决策机制
"""

import asyncio
import random
from typing import Dict, List, Any, Optional

# Import SandGraph Core components
try:
    from sandgraph.core.llm_interface import create_shared_llm_manager, create_lora_llm_manager
    from sandgraph.core.llm_frozen_adaptive import create_frozen_adaptive_llm, UpdateStrategy
    from sandgraph.core.lora_compression import create_online_lora_manager
    from sandgraph.core.rl_algorithms import RLTrainer, RLConfig, RLAlgorithm
    from sandgraph.core.reward_based_slot_manager import RewardBasedSlotManager, SlotConfig
    from sandgraph.core.monitoring import MonitoringConfig, SocialNetworkMetrics
    from sandgraph.core.sandbox import Sandbox
    from sandgraph.core.sg_workflow import SG_Workflow, WorkflowMode, NodeType
    SANGRAPH_AVAILABLE = True
except ImportError as e:
    print(f"Warning: SandGraph Core components not available: {e}")
    SANGRAPH_AVAILABLE = False

class LLMPolicy:
    """
    智能决策系统：支持frozen（只用LLM）、adaptive（RL微调）、lora（LoRA权重可插拔微调）三种模式，全部调用SandGraph core。
    集成 SandGraph Core 组件，支持更复杂的决策机制。
    """
    def __init__(self, mode='frozen', reward_fn=None, model_name="qwen-2", backend="vllm", 
                 url="http://localhost:8001/v1", lora_path=None, enable_monitoring=True):
        self.mode = mode
        self.reward_fn = reward_fn
        self.lora_path = lora_path
        self.enable_monitoring = enable_monitoring
        
        # SandGraph Core 组件
        self.llm_manager = None
        self.frozen_adaptive_llm = None
        self.lora_manager = None
        self.rl_trainer = None
        self.slot_manager = None
        self.monitoring_config = None
        
        # 初始化组件
        self._initialize_components(model_name, backend, url)
    
    def _initialize_components(self, model_name, backend, url):
        """初始化 SandGraph Core 组件"""
        if not SANGRAPH_AVAILABLE:
            print("SandGraph Core not available, using basic LLM policy")
            return
            
        try:
            if self.mode == 'frozen':
                self.llm_manager = create_shared_llm_manager(
                    model_name=model_name,
                    backend=backend,
                    url=url,
                    temperature=0.7
                )
                self.rl_trainer = None
            elif self.mode == 'adaptive':
                self.llm_manager = create_shared_llm_manager(
                    model_name=model_name,
                    backend=backend,
                    url=url,
                    temperature=0.7
                )
                rl_config = RLConfig(algorithm=RLAlgorithm.PPO)
                self.rl_trainer = RLTrainer(rl_config, self.llm_manager)
            elif self.mode == 'lora':
                self.llm_manager = create_lora_llm_manager(
                    model_name=model_name,
                    lora_path=self.lora_path,
                    backend=backend,
                    url=url,
                    temperature=0.7
                )
                rl_config = RLConfig(algorithm=RLAlgorithm.PPO)
                self.rl_trainer = RLTrainer(rl_config, self.llm_manager)
            else:
                raise ValueError(f"Unknown mode: {self.mode}")
            
            # 初始化 Frozen/Adaptive LLM
            if self.llm_manager:
                self.frozen_adaptive_llm = create_frozen_adaptive_llm(
                    self.llm_manager, 
                    strategy=UpdateStrategy.ADAPTIVE
                )
            
            # 初始化 LoRA Manager
            self.lora_manager = create_online_lora_manager(
                compression_type='hybrid',
                lora_config='medium',
                enable_online_adaptation=True
            )
            
            # 初始化 Slot Manager
            slot_config = SlotConfig(max_slots=10)
            self.slot_manager = RewardBasedSlotManager(slot_config)
            
            # 初始化监控配置
            if self.enable_monitoring:
                self.monitoring_config = MonitoringConfig(
                    enable_social_network_metrics=True,
                    enable_belief_tracking=True,
                    enable_influence_analysis=True
                )
                
        except Exception as e:
            print(f"Error initializing SandGraph Core components: {e}")
            print("Using fallback LLM policy")
    
    def _generate_enhanced_prompt(self, agent_id: int, prompt: str, state: Dict[str, Any]) -> str:
        """生成增强的 prompt"""
        if not state or "agent_states" not in state:
            return prompt
        
        agent_states = state.get("agent_states", {})
        agent_state = agent_states.get(str(agent_id), {})
        
        # 构建增强的 prompt
        enhanced_prompt = f"""
{prompt}

Additional Context:
- Agent ID: {agent_id}
- Belief Strength: {agent_state.get('belief_strength', 0.5):.2f}
- Influence Score: {agent_state.get('influence_score', 0.5):.2f}
- Neighbors Count: {agent_state.get('neighbors_count', 0)}
- Posts Count: {agent_state.get('posts_count', 0)}
- Interactions Count: {agent_state.get('interactions_count', 0)}

Consider your belief strength and influence when making your decision. 
If your belief strength is high, you're more likely to stick to your current belief.
If your influence score is high, you can have more impact on your neighbors.
"""
        return enhanced_prompt
    
    def _parse_llm_response(self, response: str) -> str:
        """解析 LLM 响应"""
        response_upper = response.upper()
        
        # 更复杂的响应解析
        if "TRUMP" in response_upper and "BIDEN" not in response_upper:
            return "TRUMP"
        elif "BIDEN" in response_upper and "TRUMP" not in response_upper:
            return "BIDEN"
        elif "TRUMP" in response_upper and "BIDEN" in response_upper:
            # 如果同时提到两者，根据上下文判断
            trump_count = response_upper.count("TRUMP")
            biden_count = response_upper.count("BIDEN")
            return "TRUMP" if trump_count > biden_count else "BIDEN"
        else:
            # 默认行为
            return random.choice(["TRUMP", "BIDEN"])
    
    def _calculate_decision_confidence(self, agent_id: int, prompt: str, state: Dict[str, Any]) -> float:
        """计算决策置信度"""
        if not state or "agent_states" not in state:
            return 0.5
        
        agent_states = state.get("agent_states", {})
        agent_state = agent_states.get(str(agent_id), {})
        
        # 基于信仰强度和影响力计算置信度
        belief_strength = agent_state.get('belief_strength', 0.5)
        influence_score = agent_state.get('influence_score', 0.5)
        
        # 信仰强度越高，置信度越高
        confidence = belief_strength * 0.6 + influence_score * 0.4
        return min(1.0, max(0.1, confidence))
    
    async def decide_async(self, prompts: Dict[int, str], state: Optional[Dict[str, Any]] = None) -> Dict[int, str]:
        """异步决策"""
        actions = {}
        
        if self.mode == 'frozen':
            for agent_id, prompt in prompts.items():
                try:
                    # 生成增强的 prompt
                    enhanced_prompt = self._generate_enhanced_prompt(agent_id, prompt, state or {})
                    
                    # 使用 LLM 生成响应
                    if self.frozen_adaptive_llm:
                        response = await self.frozen_adaptive_llm.generate(enhanced_prompt)
                    elif self.llm_manager:
                        response = self.llm_manager.run(enhanced_prompt)
                    else:
                        response = self._generate_fallback_response(enhanced_prompt)
                    
                    # 解析响应
                    action = self._parse_llm_response(response)
                    actions[agent_id] = action
                    
                    # 计算置信度
                    confidence = self._calculate_decision_confidence(agent_id, prompt, state or {})
                    
                    print(f"[LLM][Agent {agent_id}] Output: {response[:100]}... Action: {action} Confidence: {confidence:.2f}")
                    
                except Exception as e:
                    print(f"Error generating response for agent {agent_id}: {e}")
                    actions[agent_id] = self._generate_fallback_response(prompt)
                    
        elif self.mode in ('adaptive', 'lora'):
            if self.rl_trainer:
                actions = self.rl_trainer.act(prompts, state)
            else:
                # Fallback to frozen mode
                actions = await self.decide_async(prompts, state)
        
        return actions
    
    def decide(self, prompts: Dict[int, str], state: Optional[Dict[str, Any]] = None) -> Dict[int, str]:
        """
        prompts: {agent_id: prompt}
        返回: {agent_id: action}
        """
        # 同步版本的决策，内部调用异步版本
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果已经在事件循环中，直接运行
                return asyncio.run(self.decide_async(prompts, state))
            else:
                # 否则使用当前事件循环
                return loop.run_until_complete(self.decide_async(prompts, state))
        except RuntimeError:
            # 如果没有事件循环，创建新的
            return asyncio.run(self.decide_async(prompts, state))
    
    def _generate_fallback_response(self, prompt: str) -> str:
        """生成 fallback 响应"""
        # 简单的基于关键词的响应生成
        if "TRUMP" in prompt.upper():
            return "TRUMP"
        elif "BIDEN" in prompt.upper():
            return "BIDEN"
        else:
            return random.choice(["TRUMP", "BIDEN"])
    
    def update_weights(self):
        """更新权重"""
        if self.rl_trainer:
            try:
                self.rl_trainer.update_policy()
                print("RL weights updated successfully")
            except Exception as e:
                print(f"Error updating RL weights: {e}")
    
    def get_policy_info(self) -> Dict[str, Any]:
        """获取策略信息"""
        return {
            "mode": self.mode,
            "sandgraph_available": SANGRAPH_AVAILABLE,
            "llm_manager_available": self.llm_manager is not None,
            "rl_trainer_available": self.rl_trainer is not None,
            "frozen_adaptive_available": self.frozen_adaptive_llm is not None,
            "lora_manager_available": self.lora_manager is not None,
            "slot_manager_available": self.slot_manager is not None,
            "monitoring_available": self.monitoring_config is not None
        } 