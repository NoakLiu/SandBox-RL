#!/usr/bin/env python3
"""
On-Policy RL Framework with Cooperation and Competence Factors

This module implements on-policy reinforcement learning with:
- Cooperation Factor: Controls collaborative behavior between agents
- Competence Factor: Controls individual agent capability and learning
- Multi-LoRA support for single vLLM instance
- LlamaFactory integration for LoRA parameter initialization
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import random
from abc import ABC, abstractmethod
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import deque, defaultdict

logger = logging.getLogger(__name__)

class CooperationType(Enum):
    """Cooperation types for multi-agent scenarios"""
    NONE = "none"
    TEAM_BASED = "team_based"
    SHARED_REWARDS = "shared_rewards"
    KNOWLEDGE_TRANSFER = "knowledge_transfer"
    RESOURCE_SHARING = "resource_sharing"

class CompetenceType(Enum):
    """Competence types for individual agents"""
    GENERAL = "general"
    SPECIALIZED = "specialized"
    ADAPTIVE = "adaptive"
    EXPERT = "expert"
    NOVICE = "novice"

@dataclass
class CooperationFactor:
    """Cooperation factor configuration"""
    cooperation_type: CooperationType = CooperationType.NONE
    cooperation_strength: float = 0.0  # [0.0, 1.0]
    team_size: int = 1
    shared_reward_ratio: float = 0.5  # [0.0, 1.0]
    knowledge_transfer_rate: float = 0.1  # [0.0, 1.0]
    resource_sharing_enabled: bool = False
    communication_cost: float = 0.01  # Cost of cooperation
    
    def __post_init__(self):
        """Validate cooperation factor parameters"""
        assert 0.0 <= self.cooperation_strength <= 1.0, "Cooperation strength must be in [0.0, 1.0]"
        assert 0.0 <= self.shared_reward_ratio <= 1.0, "Shared reward ratio must be in [0.0, 1.0]"
        assert 0.0 <= self.knowledge_transfer_rate <= 1.0, "Knowledge transfer rate must be in [0.0, 1.0]"
        assert self.team_size >= 1, "Team size must be >= 1"

@dataclass
class CompetenceFactor:
    """Competence factor configuration"""
    competence_type: CompetenceType = CompetenceType.GENERAL
    base_capability: float = 0.5  # [0.0, 1.0]
    learning_rate: float = 0.01  # [0.0, 1.0]
    adaptation_speed: float = 0.1  # [0.0, 1.0]
    specialization_level: float = 0.0  # [0.0, 1.0]
    experience_decay: float = 0.95  # [0.0, 1.0]
    max_capability: float = 1.0  # [0.0, 1.0]
    
    def __post_init__(self):
        """Validate competence factor parameters"""
        assert 0.0 <= self.base_capability <= 1.0, "Base capability must be in [0.0, 1.0]"
        assert 0.0 <= self.learning_rate <= 1.0, "Learning rate must be in [0.0, 1.0]"
        assert 0.0 <= self.adaptation_speed <= 1.0, "Adaptation speed must be in [0.0, 1.0]"
        assert 0.0 <= self.specialization_level <= 1.0, "Specialization level must be in [0.0, 1.0]"
        assert 0.0 <= self.experience_decay <= 1.0, "Experience decay must be in [0.0, 1.0]"
        assert 0.0 <= self.max_capability <= 1.0, "Max capability must be in [0.0, 1.0]"

@dataclass
class LoRAConfig:
    """LoRA configuration for vLLM integration"""
    adapter_id: str
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    
    # LlamaFactory integration
    llama_factory_config_path: Optional[str] = None
    pretrained_model_name: Optional[str] = None
    output_dir: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for vLLM"""
        return {
            "adapter_id": self.adapter_id,
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": self.target_modules,
            "bias": self.bias,
            "task_type": self.task_type
        }

class PolicyNetwork(nn.Module):
    """Policy network for on-policy RL"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 cooperation_factor: CooperationFactor, competence_factor: CompetenceFactor):
        super().__init__()
        self.cooperation_factor = cooperation_factor
        self.competence_factor = competence_factor
        
        # Main policy network
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Cooperation-aware layers
        if cooperation_factor.cooperation_type != CooperationType.NONE:
            self.cooperation_layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim)
            )
        
        # Competence-aware layers
        self.competence_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # Output layers
        self.policy_head = nn.Linear(hidden_dim, output_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, state: torch.Tensor, cooperation_context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with cooperation and competence factors"""
        features = self.feature_extractor(state)
        
        # Apply cooperation factor
        if self.cooperation_factor.cooperation_type != CooperationType.NONE and cooperation_context is not None:
            cooperation_features = self.cooperation_layer(features)
            features = features + self.cooperation_factor.cooperation_strength * cooperation_features
        
        # Apply competence factor
        competence_features = self.competence_layer(features)
        competence_weight = self.competence_factor.base_capability + self.competence_factor.specialization_level
        features = features + competence_weight * competence_features
        
        # Output heads
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        
        return policy_logits, value

class OnPolicyRLAgent:
    """On-policy RL agent with cooperation and competence factors"""
    
    def __init__(self, 
                 agent_id: str,
                 state_dim: int,
                 action_dim: int,
                 cooperation_factor: CooperationFactor,
                 competence_factor: CompetenceFactor,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_ratio: float = 0.2,
                 value_loss_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 device: str = "cpu"):
        
        self.agent_id = agent_id
        self.device = device
        self.cooperation_factor = cooperation_factor
        self.competence_factor = competence_factor
        
        # RL parameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Networks
        self.policy_network = PolicyNetwork(
            state_dim, 256, action_dim, cooperation_factor, competence_factor
        ).to(device)
        
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        
        # Experience buffer
        self.experience_buffer = deque(maxlen=10000)
        self.episode_rewards = deque(maxlen=100)
        
        # Agent state
        self.current_capability = competence_factor.base_capability
        self.experience_count = 0
        self.team_rewards = defaultdict(float)
        
        logger.info(f"Initialized OnPolicyRLAgent {agent_id} with cooperation={cooperation_factor.cooperation_type}, competence={competence_factor.competence_type}")
    
    def get_action(self, state: np.ndarray, cooperation_context: Optional[np.ndarray] = None) -> Tuple[int, float, float]:
        """Get action from current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        cooperation_tensor = None
        if cooperation_context is not None:
            cooperation_tensor = torch.FloatTensor(cooperation_context).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            policy_logits, value = self.policy_network(state_tensor, cooperation_tensor)
            action_probs = torch.softmax(policy_logits, dim=-1)
            action_dist = Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
    
    def update_capability(self, reward: float, team_performance: Optional[float] = None):
        """Update agent capability based on performance"""
        # Individual learning
        learning_gain = self.competence_factor.learning_rate * reward
        self.current_capability += learning_gain
        
        # Team-based learning (if cooperation enabled)
        if (self.cooperation_factor.cooperation_type != CooperationType.NONE and 
            team_performance is not None):
            team_gain = self.cooperation_factor.cooperation_strength * team_performance
            self.current_capability += team_gain
        
        # Capability bounds
        self.current_capability = np.clip(
            self.current_capability, 
            0.0, 
            self.competence_factor.max_capability
        )
        
        # Experience decay
        self.current_capability *= self.competence_factor.experience_decay
        
        self.experience_count += 1
    
    def store_experience(self, state: np.ndarray, action: int, reward: float, 
                        next_state: np.ndarray, done: bool, log_prob: float, 
                        value: float, cooperation_context: Optional[np.ndarray] = None):
        """Store experience in buffer"""
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'log_prob': log_prob,
            'value': value,
            'cooperation_context': cooperation_context
        }
        self.experience_buffer.append(experience)
    
    def update_policy(self, batch_size: int = 64, num_epochs: int = 4):
        """Update policy using on-policy RL"""
        if len(self.experience_buffer) < batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.experience_buffer, batch_size)
        
        states = torch.FloatTensor([exp['state'] for exp in batch]).to(self.device)
        actions = torch.LongTensor([exp['action'] for exp in batch]).to(self.device)
        old_log_probs = torch.FloatTensor([exp['log_prob'] for exp in batch]).to(self.device)
        returns = torch.FloatTensor([exp['reward'] for exp in batch]).to(self.device)
        advantages = returns - torch.FloatTensor([exp['value'] for exp in batch]).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        cooperation_contexts = None
        if any(exp['cooperation_context'] is not None for exp in batch):
            cooperation_contexts = torch.FloatTensor([
                exp['cooperation_context'] if exp['cooperation_context'] is not None 
                else np.zeros(states.shape[1]) for exp in batch
            ]).to(self.device)
        
        for epoch in range(num_epochs):
            # Forward pass
            policy_logits, values = self.policy_network(states, cooperation_contexts)
            action_probs = torch.softmax(policy_logits, dim=-1)
            action_dist = Categorical(action_probs)
            log_probs = action_dist.log_prob(actions)
            
            # Compute losses
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            value_loss = nn.MSELoss()(values.squeeze(), returns)
            entropy_loss = -action_dist.entropy().mean()
            
            total_loss = (policy_loss + 
                         self.value_loss_coef * value_loss + 
                         self.entropy_coef * entropy_loss)
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.max_grad_norm)
            self.optimizer.step()
        
        # Clear buffer after update
        self.experience_buffer.clear()
        
        logger.info(f"Agent {self.agent_id} policy updated. Loss: {total_loss.item():.4f}")

class MultiLoRAOnPolicyRL:
    """Multi-LoRA on-policy RL system for single vLLM instance"""
    
    def __init__(self, 
                 num_adapters: int = 8,
                 state_dim: int = 64,
                 action_dim: int = 10,
                 cooperation_configs: Optional[List[CooperationFactor]] = None,
                 competence_configs: Optional[List[CompetenceFactor]] = None,
                 device: str = "cpu"):
        
        self.num_adapters = num_adapters
        self.device = device
        self.agents = {}
        self.lora_configs = {}
        
        # Initialize cooperation and competence configs
        if cooperation_configs is None:
            cooperation_configs = [CooperationFactor() for _ in range(num_adapters)]
        if competence_configs is None:
            competence_configs = [CompetenceFactor() for _ in range(num_adapters)]
        
        # Create agents for each LoRA adapter
        for i in range(num_adapters):
            agent_id = f"lora_agent_{i}"
            self.agents[agent_id] = OnPolicyRLAgent(
                agent_id=agent_id,
                state_dim=state_dim,
                action_dim=action_dim,
                cooperation_factor=cooperation_configs[i],
                competence_factor=competence_configs[i],
                device=device
            )
            
            # Create LoRA config
            self.lora_configs[agent_id] = LoRAConfig(
                adapter_id=f"lora_adapter_{i}",
                rank=16,
                alpha=32.0
            )
        
        # Team management
        self.teams = self._create_teams()
        self.team_performance = defaultdict(float)
        
        logger.info(f"Initialized MultiLoRAOnPolicyRL with {num_adapters} adapters")
    
    def _create_teams(self) -> Dict[str, List[str]]:
        """Create teams based on cooperation factors"""
        teams = {}
        team_id = 0
        
        for agent_id, agent in self.agents.items():
            if agent.cooperation_factor.cooperation_type == CooperationType.TEAM_BASED:
                team_key = f"team_{team_id // agent.cooperation_factor.team_size}"
                if team_key not in teams:
                    teams[team_key] = []
                teams[team_key].append(agent_id)
                team_id += 1
            else:
                # Individual agent
                teams[f"individual_{agent_id}"] = [agent_id]
        
        return teams
    
    def get_cooperation_context(self, agent_id: str, state: np.ndarray) -> Optional[np.ndarray]:
        """Get cooperation context for an agent"""
        agent = self.agents[agent_id]
        
        if agent.cooperation_factor.cooperation_type == CooperationType.NONE:
            return None
        
        # Find team members
        team_members = []
        for team_id, members in self.teams.items():
            if agent_id in members:
                team_members = members
                break
        
        if len(team_members) <= 1:
            return None
        
        # Aggregate team information
        team_states = []
        for member_id in team_members:
            if member_id != agent_id:
                # Use current state as approximation for other agents
                team_states.append(state)
        
        if team_states:
            team_context = np.mean(team_states, axis=0)
            return team_context * agent.cooperation_factor.cooperation_strength
        
        return None
    
    def step(self, adapter_id: str, state: np.ndarray) -> Tuple[int, float, float]:
        """Take a step with a specific LoRA adapter"""
        agent_id = f"lora_agent_{adapter_id}"
        if agent_id not in self.agents:
            raise ValueError(f"Unknown adapter ID: {adapter_id}")
        
        agent = self.agents[agent_id]
        cooperation_context = self.get_cooperation_context(agent_id, state)
        
        action, log_prob, value = agent.get_action(state, cooperation_context)
        
        return action, log_prob, value
    
    def update_agent(self, adapter_id: str, state: np.ndarray, action: int, 
                    reward: float, next_state: np.ndarray, done: bool, 
                    log_prob: float, value: float):
        """Update agent experience and policy"""
        agent_id = f"lora_agent_{adapter_id}"
        agent = self.agents[agent_id]
        
        # Get team performance for cooperation
        team_performance = None
        if agent.cooperation_factor.cooperation_type != CooperationType.NONE:
            team_performance = self.team_performance.get(f"team_{adapter_id}", 0.0)
        
        # Update capability
        agent.update_capability(reward, team_performance)
        
        # Store experience
        cooperation_context = self.get_cooperation_context(agent_id, state)
        agent.store_experience(state, action, reward, next_state, done, log_prob, value, cooperation_context)
        
        # Update team performance
        if agent.cooperation_factor.cooperation_type != CooperationType.NONE:
            self.team_performance[f"team_{adapter_id}"] = reward
    
    def update_all_policies(self, batch_size: int = 64, num_epochs: int = 4):
        """Update policies for all agents"""
        for agent_id, agent in self.agents.items():
            agent.update_policy(batch_size, num_epochs)
    
    def get_lora_configs(self) -> Dict[str, LoRAConfig]:
        """Get LoRA configurations for vLLM"""
        return self.lora_configs
    
    def get_agent_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all agents"""
        stats = {}
        for agent_id, agent in self.agents.items():
            stats[agent_id] = {
                'capability': agent.current_capability,
                'experience_count': agent.experience_count,
                'cooperation_type': agent.cooperation_factor.cooperation_type.value,
                'competence_type': agent.competence_factor.competence_type.value,
                'avg_reward': np.mean(agent.episode_rewards) if agent.episode_rewards else 0.0
            }
        return stats

def create_on_policy_rl_system(num_adapters: int = 8,
                              cooperation_type: CooperationType = CooperationType.TEAM_BASED,
                              competence_type: CompetenceType = CompetenceType.ADAPTIVE,
                              device: str = "cpu") -> MultiLoRAOnPolicyRL:
    """Create on-policy RL system with cooperation and competence factors"""
    
    # Create cooperation configs
    cooperation_configs = []
    for i in range(num_adapters):
        if cooperation_type == CooperationType.TEAM_BASED:
            team_size = min(4, num_adapters // 2)
            cooperation_configs.append(CooperationFactor(
                cooperation_type=CooperationType.TEAM_BASED,
                cooperation_strength=0.3 + 0.1 * (i % 2),  # Vary cooperation strength
                team_size=team_size,
                shared_reward_ratio=0.6
            ))
        else:
            cooperation_configs.append(CooperationFactor(
                cooperation_type=cooperation_type,
                cooperation_strength=0.2
            ))
    
    # Create competence configs
    competence_configs = []
    for i in range(num_adapters):
        if competence_type == CompetenceType.ADAPTIVE:
            competence_configs.append(CompetenceFactor(
                competence_type=CompetenceType.ADAPTIVE,
                base_capability=0.4 + 0.1 * (i % 3),  # Vary base capability
                learning_rate=0.02 + 0.01 * (i % 2),
                adaptation_speed=0.15 + 0.05 * (i % 2)
            ))
        else:
            competence_configs.append(CompetenceFactor(
                competence_type=competence_type,
                base_capability=0.5 + 0.1 * (i % 2)
            ))
    
    return MultiLoRAOnPolicyRL(
        num_adapters=num_adapters,
        cooperation_configs=cooperation_configs,
        competence_configs=competence_configs,
        device=device
    )

def initialize_vllm_lora_adapters(llama_factory_config_path: str, 
                                pretrained_model_name: str,
                                output_dir: str,
                                num_adapters: int = 8) -> Dict[str, LoRAConfig]:
    """Initialize vLLM LoRA adapters with LlamaFactory parameters"""
    
    lora_configs = {}
    
    for i in range(num_adapters):
        adapter_id = f"lora_adapter_{i}"
        lora_configs[adapter_id] = LoRAConfig(
            adapter_id=adapter_id,
            rank=16,
            alpha=32.0,
            dropout=0.1,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM",
            llama_factory_config_path=llama_factory_config_path,
            pretrained_model_name=pretrained_model_name,
            output_dir=f"{output_dir}/adapter_{i}"
        )
    
    logger.info(f"Initialized {num_adapters} LoRA adapters with LlamaFactory config")
    return lora_configs
