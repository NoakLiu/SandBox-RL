#!/usr/bin/env python3
"""
LoRA RL Manager - On-Policy Reinforcement Learning for LoRA Management

This module provides:
1. On-policy RL algorithms for LoRA weight updates
2. LlamaFactory integration for initializing vLLM LoRA adapters
3. Policy gradient methods for adaptive LoRA optimization
4. Experience replay and policy improvement mechanisms

Architecture: Single vLLM + 8 LoRA adapters with RL-driven weight updates
"""

import asyncio
import time
import random
import logging
import json
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import aiohttp
import requests
import os
import shutil

logger = logging.getLogger(__name__)


class RLAlgorithm(Enum):
    """Reinforcement Learning algorithms"""
    POLICY_GRADIENT = "policy_gradient"
    ACTOR_CRITIC = "actor_critic"
    PPO = "ppo"  # Proximal Policy Optimization
    A2C = "a2c"  # Advantage Actor-Critic


class LoRAUpdateAction(Enum):
    """LoRA update actions"""
    INCREASE_RANK = "increase_rank"
    DECREASE_RANK = "decrease_rank"
    ADJUST_ALPHA = "adjust_alpha"
    CHANGE_DROPOUT = "change_dropout"
    FREEZE_LAYERS = "freeze_layers"
    UNFREEZE_LAYERS = "unfreeze_layers"
    MAINTAIN_CURRENT = "maintain_current"


@dataclass
class LoRAConfig:
    """LoRA configuration parameters"""
    r: int = 16                    # LoRA rank
    lora_alpha: int = 32           # LoRA alpha parameter
    lora_dropout: float = 0.1      # LoRA dropout rate
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"             # Bias handling
    task_type: str = "CAUSAL_LM"   # Task type
    inference_mode: bool = False   # Inference mode flag
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for LlamaFactory"""
        return {
            'r': self.r,
            'lora_alpha': self.lora_alpha,
            'lora_dropout': self.lora_dropout,
            'target_modules': self.target_modules,
            'bias': self.bias,
            'task_type': self.task_type,
            'inference_mode': self.inference_mode
        }


@dataclass
class LoRAState:
    """LoRA state representation"""
    adapter_id: int
    current_config: LoRAConfig
    performance_history: List[float] = field(default_factory=list)
    reward_history: List[float] = field(default_factory=list)
    action_history: List[LoRAUpdateAction] = field(default_factory=list)
    last_update_time: float = field(default_factory=time.time)
    exploration_rate: float = 0.1
    success_rate: float = 0.5
    
    def get_state_vector(self) -> np.ndarray:
        """Get state vector for RL agent"""
        return np.array([
            self.current_config.r / 64.0,  # Normalized rank
            self.current_config.lora_alpha / 64.0,  # Normalized alpha
            self.current_config.lora_dropout,
            self.exploration_rate,
            self.success_rate,
            len(self.performance_history) / 100.0,  # Normalized history length
            np.mean(self.performance_history[-10:]) if self.performance_history else 0.5,  # Recent performance
            np.std(self.performance_history[-10:]) if len(self.performance_history) >= 10 else 0.1  # Performance stability
        ])


@dataclass
class RLExperience:
    """RL experience for training"""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    timestamp: float = field(default_factory=time.time)


class PolicyNetwork(nn.Module):
    """Policy network for LoRA update decisions"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(PolicyNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.softmax(x, dim=-1)


class ValueNetwork(nn.Module):
    """Value network for critic (used in Actor-Critic methods)"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super(ValueNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        self.dropout = nn.Dropout(0.1)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class OnPolicyRLAgent:
    """On-policy RL agent for LoRA management"""
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 algorithm: RLAlgorithm = RLAlgorithm.PPO,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_ratio: float = 0.2,
                 value_loss_coef: float = 0.5,
                 entropy_coef: float = 0.01):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.algorithm = algorithm
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        
        # Networks
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.value_net = ValueNetwork(state_dim)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)
        
        # Experience buffer
        self.experience_buffer = []
        self.max_buffer_size = 10000
        
        # Training parameters
        self.batch_size = 64
        self.update_frequency = 100  # Update every N experiences
        
        # Statistics
        self.training_steps = 0
        self.total_reward = 0.0
        self.episode_count = 0
        
        logger.info(f"Initialized {algorithm.value} RL agent")
    
    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, float]:
        """Select action using current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action_probs = self.policy_net(state_tensor)
            action_dist = Categorical(action_probs)
            
            if training:
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
            else:
                action = torch.argmax(action_probs)
                log_prob = torch.tensor(0.0)
        
        return action.item(), log_prob.item()
    
    def store_experience(self, experience: RLExperience):
        """Store experience in buffer"""
        self.experience_buffer.append(experience)
        
        # Maintain buffer size
        if len(self.experience_buffer) > self.max_buffer_size:
            self.experience_buffer.pop(0)
        
        # Check if we should update
        if len(self.experience_buffer) >= self.batch_size and len(self.experience_buffer) % self.update_frequency == 0:
            self._update_policy()
    
    def _update_policy(self):
        """Update policy using stored experiences"""
        if len(self.experience_buffer) < self.batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.experience_buffer, self.batch_size)
        
        # Prepare batch data
        states = torch.FloatTensor([exp.state for exp in batch])
        actions = torch.LongTensor([exp.action for exp in batch])
        rewards = torch.FloatTensor([exp.reward for exp in batch])
        next_states = torch.FloatTensor([exp.next_state for exp in batch])
        dones = torch.BoolTensor([exp.done for exp in batch])
        
        # Calculate advantages using GAE
        advantages = self._calculate_gae(states, rewards, next_states, dones)
        
        if self.algorithm == RLAlgorithm.PPO:
            self._update_ppo(states, actions, advantages, rewards)
        elif self.algorithm == RLAlgorithm.ACTOR_CRITIC:
            self._update_actor_critic(states, actions, advantages, rewards)
        elif self.algorithm == RLAlgorithm.POLICY_GRADIENT:
            self._update_policy_gradient(states, actions, advantages)
        
        self.training_steps += 1
        
        # Clear buffer after update
        self.experience_buffer.clear()
    
    def _calculate_gae(self, states: torch.Tensor, rewards: torch.Tensor,
                      next_states: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """Calculate Generalized Advantage Estimation"""
        with torch.no_grad():
            values = self.value_net(states).squeeze()
            next_values = self.value_net(next_states).squeeze()
            
            # Handle terminal states
            next_values = torch.where(dones, torch.zeros_like(next_values), next_values)
            
            # Calculate TD errors
            deltas = rewards + self.gamma * next_values - values
            
            # Calculate GAE
            advantages = torch.zeros_like(rewards)
            gae = 0.0
            
            for t in reversed(range(len(rewards))):
                gae = deltas[t] + self.gamma * self.gae_lambda * gae * (1 - dones[t])
                advantages[t] = gae
        
        return advantages
    
    def _update_ppo(self, states: torch.Tensor, actions: torch.Tensor,
                   advantages: torch.Tensor, rewards: torch.Tensor):
        """Update using PPO algorithm"""
        # Get current action probabilities
        action_probs = self.policy_net(states)
        action_dist = Categorical(action_probs)
        current_log_probs = action_dist.log_prob(actions)
        
        # Calculate policy loss with clipping
        ratio = torch.exp(current_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        values = self.value_net(states).squeeze()
        value_loss = F.mse_loss(values, rewards)
        
        # Entropy bonus
        entropy = action_dist.entropy().mean()
        
        # Total loss
        total_loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
        
        # Update networks
        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        total_loss.backward()
        self.policy_optimizer.step()
        self.value_optimizer.step()
        
        logger.debug(f"PPO update - Policy loss: {policy_loss.item():.4f}, "
                    f"Value loss: {value_loss.item():.4f}, Entropy: {entropy.item():.4f}")
    
    def _update_actor_critic(self, states: torch.Tensor, actions: torch.Tensor,
                            advantages: torch.Tensor, rewards: torch.Tensor):
        """Update using Actor-Critic algorithm"""
        # Policy loss
        action_probs = self.policy_net(states)
        action_dist = Categorical(action_probs)
        log_probs = action_dist.log_prob(actions)
        policy_loss = -(log_probs * advantages.detach()).mean()
        
        # Value loss
        values = self.value_net(states).squeeze()
        value_loss = F.mse_loss(values, rewards)
        
        # Update networks
        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        
        policy_loss.backward()
        value_loss.backward()
        
        self.policy_optimizer.step()
        self.value_optimizer.step()
        
        logger.debug(f"Actor-Critic update - Policy loss: {policy_loss.item():.4f}, "
                    f"Value loss: {value_loss.item():.4f}")
    
    def _update_policy_gradient(self, states: torch.Tensor, actions: torch.Tensor,
                               advantages: torch.Tensor):
        """Update using vanilla Policy Gradient"""
        action_probs = self.policy_net(states)
        action_dist = Categorical(action_probs)
        log_probs = action_dist.log_prob(actions)
        
        policy_loss = -(log_probs * advantages.detach()).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        logger.debug(f"Policy Gradient update - Loss: {policy_loss.item():.4f}")
    
    def save_model(self, path: str):
        """Save trained model"""
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'value_state_dict': self.value_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'training_steps': self.training_steps,
            'total_reward': self.total_reward,
            'episode_count': self.episode_count
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        self.training_steps = checkpoint['training_steps']
        self.total_reward = checkpoint['total_reward']
        self.episode_count = checkpoint['episode_count']
        logger.info(f"Model loaded from {path}")


class LoRARLManager:
    """Main RL manager for LoRA updates"""
    
    def __init__(self, 
                 vllm_server_url: str = "http://localhost:8001",
                 lora_base_path: str = "/cpfs04/shared/kilab/liudong",
                 algorithm: RLAlgorithm = RLAlgorithm.PPO):
        
        self.vllm_server_url = vllm_server_url
        self.lora_base_path = lora_base_path
        self.algorithm = algorithm
        
        # Initialize RL agent
        state_dim = 8  # LoRAState.get_state_vector() returns 8-dimensional vector
        action_dim = len(LoRAUpdateAction)
        
        self.rl_agent = OnPolicyRLAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            algorithm=algorithm
        )
        
        # LoRA states
        self.lora_states: Dict[int, LoRAState] = {}
        self._initialize_lora_states()
        
        # Performance tracking
        self.performance_history = []
        self.update_history = []
        
        # Configuration
        self.update_interval = 60  # Update every 60 seconds
        self.min_experiences_for_update = 10
        
        logger.info(f"LoRA RL Manager initialized with {algorithm.value} algorithm")
    
    def _initialize_lora_states(self):
        """Initialize LoRA states for 8 adapters"""
        for i in range(1, 9):
            config = LoRAConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj"]
            )
            
            self.lora_states[i] = LoRAState(
                adapter_id=i,
                current_config=config
            )
    
    async def run_rl_cycle(self, num_episodes: int = 100) -> Dict[str, Any]:
        """Run RL training cycle"""
        logger.info(f"Starting RL training cycle with {num_episodes} episodes")
        
        results = {
            'algorithm': self.algorithm.value,
            'episodes': [],
            'final_statistics': {}
        }
        
        for episode in range(num_episodes):
            logger.info(f"Episode {episode + 1}/{num_episodes}")
            
            # Run episode
            episode_result = await self._run_episode(episode)
            results['episodes'].append(episode_result)
            
            # Update RL agent if enough experiences
            if len(self.rl_agent.experience_buffer) >= self.min_experiences_for_update:
                self.rl_agent._update_policy()
            
            # Log progress
            avg_reward = np.mean([exp.reward for exp in self.rl_agent.experience_buffer[-100:]])
            logger.info(f"Episode {episode + 1} complete - Avg reward: {avg_reward:.3f}")
            
            # Small delay between episodes
            await asyncio.sleep(1)
        
        # Final statistics
        results['final_statistics'] = self._get_rl_statistics()
        
        logger.info(f"RL training cycle complete")
        return results
    
    async def _run_episode(self, episode_num: int) -> Dict[str, Any]:
        """Run a single RL episode"""
        episode_reward = 0.0
        episode_actions = []
        
        # Run for each LoRA adapter
        for adapter_id, lora_state in self.lora_states.items():
            # Get current state
            current_state = lora_state.get_state_vector()
            
            # Select action
            action_idx, log_prob = self.rl_agent.select_action(current_state, training=True)
            action = list(LoRAUpdateAction)[action_idx]
            
            # Execute action
            reward = await self._execute_lora_action(adapter_id, action, lora_state)
            
            # Get next state
            next_state = lora_state.get_state_vector()
            
            # Store experience
            experience = RLExperience(
                state=current_state,
                action=action_idx,
                reward=reward,
                next_state=next_state,
                done=False
            )
            self.rl_agent.store_experience(experience)
            
            episode_reward += reward
            episode_actions.append({
                'adapter_id': adapter_id,
                'action': action.value,
                'reward': reward,
                'log_prob': log_prob
            })
            
            # Update LoRA state
            lora_state.reward_history.append(reward)
            lora_state.action_history.append(action)
            lora_state.last_update_time = time.time()
            
            # Update success rate
            if reward > 0.5:
                lora_state.success_rate = lora_state.success_rate * 0.9 + 0.1
            else:
                lora_state.success_rate = lora_state.success_rate * 0.9 + 0.0
        
        # Record episode
        episode_result = {
            'episode': episode_num + 1,
            'total_reward': episode_reward,
            'actions': episode_actions,
            'average_reward': episode_reward / len(self.lora_states)
        }
        
        self.performance_history.append(episode_result)
        self.rl_agent.episode_count += 1
        self.rl_agent.total_reward += episode_reward
        
        return episode_result
    
    async def _execute_lora_action(self, adapter_id: int, action: LoRAUpdateAction, 
                                 lora_state: LoRAState) -> float:
        """Execute LoRA update action and return reward"""
        try:
            old_config = lora_state.current_config
            
            if action == LoRAUpdateAction.INCREASE_RANK:
                lora_state.current_config.r = min(64, lora_state.current_config.r + 4)
            elif action == LoRAUpdateAction.DECREASE_RANK:
                lora_state.current_config.r = max(4, lora_state.current_config.r - 4)
            elif action == LoRAUpdateAction.ADJUST_ALPHA:
                lora_state.current_config.lora_alpha = min(64, lora_state.current_config.lora_alpha + 8)
            elif action == LoRAUpdateAction.CHANGE_DROPOUT:
                lora_state.current_config.lora_dropout = max(0.0, min(0.5, 
                    lora_state.current_config.lora_dropout + random.uniform(-0.05, 0.05)))
            elif action == LoRAUpdateAction.FREEZE_LAYERS:
                # Simulate freezing some layers
                pass
            elif action == LoRAUpdateAction.UNFREEZE_LAYERS:
                # Simulate unfreezing some layers
                pass
            elif action == LoRAUpdateAction.MAINTAIN_CURRENT:
                pass
            
            # Test the new configuration
            reward = await self._evaluate_lora_config(adapter_id, lora_state.current_config)
            
            # If reward is poor, revert changes
            if reward < 0.3:
                lora_state.current_config = old_config
                reward = reward * 0.5  # Penalty for poor changes
            
            return reward
            
        except Exception as e:
            logger.error(f"Error executing action {action.value} for adapter {adapter_id}: {e}")
            return 0.1  # Minimum reward
    
    async def _evaluate_lora_config(self, adapter_id: int, config: LoRAConfig) -> float:
        """Evaluate LoRA configuration performance"""
        try:
            # Test generation with current config
            test_prompt = f"Test prompt for adapter {adapter_id} with config r={config.r}, alpha={config.lora_alpha}"
            
            # Simulate API call to vLLM
            response = await self._test_lora_generation(adapter_id, test_prompt)
            
            # Calculate reward based on response quality and config efficiency
            response_quality = min(1.0, len(response) / 50.0)
            config_efficiency = 1.0 / (config.r * config.lora_alpha / 512.0)  # Normalized efficiency
            
            # Combined reward
            reward = (response_quality * 0.7 + config_efficiency * 0.3) * 0.8 + 0.2
            
            # Add some randomness for realistic variation
            reward += random.uniform(-0.05, 0.05)
            reward = max(0.1, min(1.0, reward))
            
            return reward
            
        except Exception as e:
            logger.error(f"Error evaluating LoRA config for adapter {adapter_id}: {e}")
            return 0.1
    
    async def _test_lora_generation(self, adapter_id: int, prompt: str) -> str:
        """Test LoRA generation via vLLM API"""
        try:
            # This would be a real API call to vLLM
            # For now, return a mock response
            return f"[Mock] Adapter {adapter_id} response to: {prompt}"
            
        except Exception as e:
            logger.error(f"Error testing LoRA generation for adapter {adapter_id}: {e}")
            return f"[Error] {str(e)}"
    
    def _get_rl_statistics(self) -> Dict[str, Any]:
        """Get RL training statistics"""
        if not self.performance_history:
            return {}
        
        recent_episodes = self.performance_history[-50:]  # Last 50 episodes
        
        return {
            'algorithm': self.algorithm.value,
            'total_episodes': self.rl_agent.episode_count,
            'total_reward': self.rl_agent.total_reward,
            'average_reward_per_episode': self.rl_agent.total_reward / max(self.rl_agent.episode_count, 1),
            'recent_performance': {
                'last_10_episodes': np.mean([ep['average_reward'] for ep in recent_episodes[-10:]]),
                'last_50_episodes': np.mean([ep['average_reward'] for ep in recent_episodes]),
                'performance_trend': np.polyfit(range(len(recent_episodes)), 
                                             [ep['average_reward'] for ep in recent_episodes], 1)[0]
            },
            'lora_adapter_stats': {
                adapter_id: {
                    'success_rate': state.success_rate,
                    'total_rewards': sum(state.reward_history),
                    'action_distribution': {
                        action.value: state.action_history.count(action) / max(len(state.action_history), 1)
                        for action in LoRAUpdateAction
                    }
                }
                for adapter_id, state in self.lora_states.items()
            }
        }
    
    def save_rl_model(self, path: str):
        """Save RL model"""
        self.rl_agent.save_model(path)
    
    def load_rl_model(self, path: str):
        """Load RL model"""
        self.rl_agent.load_model(path)


# LlamaFactory Integration Functions
def create_llamafactory_lora_config(
    base_model: str = "Qwen2.5-7B-Instruct",
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    target_modules: Optional[List[str]] = None,
    bias: str = "none",
    task_type: str = "CAUSAL_LM"
) -> Dict[str, Any]:
    """Create LlamaFactory LoRA configuration"""
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]
    
    return {
        'base_model': base_model,
        'lora_config': {
            'r': lora_r,
            'lora_alpha': lora_alpha,
            'lora_dropout': lora_dropout,
            'target_modules': target_modules,
            'bias': bias,
            'task_type': task_type
        }
    }


def generate_llamafactory_training_script(
    base_model: str,
    lora_config: Dict[str, Any],
    output_dir: str = "./lora_output",
    training_data: str = "./training_data.json",
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4
) -> str:
    """Generate LlamaFactory training script"""
    
    script_content = f"""#!/bin/bash

# LlamaFactory LoRA Training Script
# Generated automatically for {base_model}

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Training parameters
BASE_MODEL="{base_model}"
OUTPUT_DIR="{output_dir}"
TRAINING_DATA="{training_data}"
NUM_EPOCHS={num_epochs}
LEARNING_RATE={learning_rate}
BATCH_SIZE={batch_size}
GRADIENT_ACCUMULATION_STEPS={gradient_accumulation_steps}

# LoRA parameters
LORA_R={lora_config['lora_config']['r']}
LORA_ALPHA={lora_config['lora_config']['lora_alpha']}
LORA_DROPOUT={lora_config['lora_config']['lora_dropout']}
TARGET_MODULES="{','.join(lora_config['lora_config']['target_modules'])}"

echo "Starting LoRA training with LlamaFactory..."
echo "Base model: $BASE_MODEL"
echo "LoRA config: r=$LORA_R, alpha=$LORA_ALPHA, dropout=$LORA_DROPOUT"

# Run LlamaFactory training
llamafactory-cli train \\
    --base_model $BASE_MODEL \\
    --output_dir $OUTPUT_DIR \\
    --train_data $TRAINING_DATA \\
    --num_epochs $NUM_EPOCHS \\
    --learning_rate $LEARNING_RATE \\
    --batch_size $BATCH_SIZE \\
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \\
    --lora_r $LORA_R \\
    --lora_alpha $LORA_ALPHA \\
    --lora_dropout $LORA_DROPOUT \\
    --target_modules $TARGET_MODULES \\
    --save_steps 100 \\
    --logging_steps 10 \\
    --evaluation_strategy "steps" \\
    --eval_steps 100 \\
    --save_total_limit 3 \\
    --load_best_model_at_end \\
    --metric_for_best_model "eval_loss" \\
    --greater_is_better false

echo "LoRA training completed!"
echo "Output saved to: $OUTPUT_DIR"
"""
    
    return script_content


def create_vllm_lora_adapter(
    lora_path: str,
    adapter_name: str,
    base_model: str = "Qwen2.5-7B-Instruct",
    max_lora_rank: int = 64,
    max_lora_modules: int = 8
) -> Dict[str, Any]:
    """Create vLLM LoRA adapter configuration"""
    
    # Check if LoRA files exist
    adapter_config_path = os.path.join(lora_path, "adapter_config.json")
    adapter_model_path = os.path.join(lora_path, "adapter_model.bin")
    
    if not os.path.exists(adapter_config_path):
        raise FileNotFoundError(f"LoRA config not found: {adapter_config_path}")
    
    if not os.path.exists(adapter_model_path):
        raise FileNotFoundError(f"LoRA model not found: {adapter_model_path}")
    
    # Read LoRA config
    with open(adapter_config_path, 'r') as f:
        lora_config = json.load(f)
    
    # Validate LoRA parameters
    if lora_config.get('r', 16) > max_lora_rank:
        raise ValueError(f"LoRA rank {lora_config['r']} exceeds maximum {max_lora_rank}")
    
    # Create vLLM adapter config
    vllm_adapter_config = {
        'adapter_name': adapter_name,
        'adapter_path': lora_path,
        'base_model': base_model,
        'lora_config': {
            'r': lora_config.get('r', 16),
            'lora_alpha': lora_config.get('lora_alpha', 32),
            'lora_dropout': lora_config.get('lora_dropout', 0.1),
            'target_modules': lora_config.get('target_modules', ["q_proj", "v_proj"]),
            'bias': lora_config.get('bias', 'none'),
            'modules_to_save': lora_config.get('modules_to_save', None)
        }
    }
    
    return vllm_adapter_config


def generate_vllm_launch_script(
    base_model: str,
    lora_adapters: List[Dict[str, Any]],
    tensor_parallel_size: int = 4,
    max_lora_rank: int = 64,
    max_lora_modules: int = 8,
    port: int = 8001
) -> str:
    """Generate vLLM launch script with LoRA adapters"""
    
    # Build LoRA modules argument
    lora_modules_arg = []
    for adapter in lora_adapters:
        lora_modules_arg.append(f"{adapter['adapter_name']}={adapter['adapter_path']}")
    
    lora_modules_str = ",".join(lora_modules_arg)
    
    script_content = f"""#!/bin/bash

# vLLM Launch Script with LoRA Adapters
# Generated automatically

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Model and LoRA parameters
BASE_MODEL="{base_model}"
LORA_MODULES="{lora_modules_str}"
TENSOR_PARALLEL_SIZE={tensor_parallel_size}
MAX_LORA_RANK={max_lora_rank}
MAX_LORA_MODULES={max_lora_modules}
PORT={port}

echo "Launching vLLM server with LoRA adapters..."
echo "Base model: $BASE_MODEL"
echo "LoRA modules: $LORA_MODULES"
echo "Tensor parallel size: $TENSOR_PARALLEL_SIZE"

# Launch vLLM server
vllm serve $BASE_MODEL \\
    --port $PORT \\
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \\
    --max-lora-rank $MAX_LORA_RANK \\
    --max-lora-modules $MAX_LORA_MODULES \\
    --lora-modules $LORA_MODULES \\
    --dtype bfloat16 \\
    --max-model-len 16384 \\
    --gpu-memory-utilization 0.8 \\
    --max-num-seqs 256 \\
    --enforce-eager

echo "vLLM server launched on port $PORT"
echo "LoRA adapters loaded: {len(lora_adapters)}"
"""
    
    return script_content


# Factory functions
def create_lora_rl_manager(
    vllm_server_url: str = "http://localhost:8001",
    algorithm: RLAlgorithm = RLAlgorithm.PPO
) -> LoRARLManager:
    """Create LoRA RL manager"""
    return LoRARLManager(vllm_server_url=vllm_server_url, algorithm=algorithm)


def create_ppo_lora_manager(vllm_server_url: str = "http://localhost:8001") -> LoRARLManager:
    """Create PPO-based LoRA RL manager"""
    return create_lora_rl_manager(vllm_server_url, RLAlgorithm.PPO)


def create_actor_critic_lora_manager(vllm_server_url: str = "http://localhost:8001") -> LoRARLManager:
    """Create Actor-Critic based LoRA RL manager"""
    return create_lora_rl_manager(vllm_server_url, RLAlgorithm.ACTOR_CRITIC)


async def main():
    """Example usage"""
    print("🚀 LoRA RL Manager Demo")
    print("=" * 50)
    
    # Create RL manager
    rl_manager = create_ppo_lora_manager()
    
    # Run RL training
    results = await rl_manager.run_rl_cycle(num_episodes=10)
    
    print(f"RL training results: {json.dumps(results['final_statistics'], indent=2)}")
    
    # Save model
    rl_manager.save_rl_model("./lora_rl_model.pth")
    
    print("\n✅ Demo complete!")


if __name__ == "__main__":
    asyncio.run(main())
