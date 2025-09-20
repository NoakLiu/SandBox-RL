#!/usr/bin/env python3
"""
Multi-Model Trainer - Core Multi-Model RL Training System
=========================================================

Advanced multi-model training system with:
1. Simultaneous training of 4-8 modern LLMs
2. Real-time weight updates and parameter synchronization
3. Cooperative-competitive RL dynamics
4. VERL/AReaL integration for efficient training
5. Checkpoint management and recovery
6. Live performance monitoring
"""

import asyncio
import logging
import time
import json
import os
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from .llm_manager import SharedLLMManager, create_qwen3_manager, create_openai_manager, create_claude_manager
from .rl_framework import RLTrainer, create_ppo_trainer, create_grpo_trainer, MultiAgentOnPolicyRL
from .lora_system import DistributedLoRAScheduler, LoRARLStrategy
from .scheduler import UnifiedScheduler, create_unified_scheduler
from .cache_optimizer import AReaLVERLBridge, VERLTrainer
from .monitoring import UnifiedMonitor, create_unified_monitor
from .environments import MultiModelTrainingEnvironment, create_multi_model_coop_compete_env

logger = logging.getLogger(__name__)


class TrainingMode(Enum):
    """Multi-model training modes"""
    COOPERATIVE = "cooperative"  # Models help each other
    COMPETITIVE = "competitive"  # Models compete for resources
    MIXED = "mixed"  # Dynamic cooperation/competition
    HIERARCHICAL = "hierarchical"  # Leader-follower dynamics


class WeightUpdateStrategy(Enum):
    """Weight update strategies for multi-model training"""
    SYNCHRONIZED = "synchronized"  # All models update together
    ASYNCHRONOUS = "asynchronous"  # Models update independently
    FEDERATED = "federated"  # Federated learning style updates
    SELECTIVE = "selective"  # Only best performing models update


@dataclass
class MultiModelConfig:
    """Configuration for multi-model training"""
    # Model configuration
    num_models: int = 4
    model_types: List[str] = field(default_factory=lambda: ["qwen3", "openai", "claude", "llama3"])
    model_names: Dict[str, str] = field(default_factory=lambda: {
        "qwen3": "Qwen/Qwen2.5-14B-Instruct",
        "openai": "gpt-4o-mini", 
        "claude": "claude-3-5-haiku-20241022",
        "llama3": "meta-llama/Llama-3.1-8B-Instruct"
    })
    
    # Training configuration
    training_mode: TrainingMode = TrainingMode.MIXED
    weight_update_strategy: WeightUpdateStrategy = WeightUpdateStrategy.ASYNCHRONOUS
    max_episodes: int = 1000
    episode_length: int = 32
    update_frequency: int = 10  # Update weights every N episodes
    
    # RL configuration
    learning_rate: float = 3e-4
    cooperation_strength: float = 0.6
    competition_intensity: float = 0.4
    
    # Resource configuration
    base_port: int = 8001
    num_gpus: int = 4
    batch_size: int = 32
    
    # Checkpoint configuration
    checkpoint_dir: str = "./checkpoints/multimodel"
    save_interval: int = 100  # Save every N episodes
    max_checkpoints: int = 10  # Keep last N checkpoints
    
    # VERL/AReaL integration
    enable_verl: bool = True
    enable_areal: bool = True
    kv_cache_size: int = 10000
    
    # Monitoring
    enable_monitoring: bool = True
    log_metrics_interval: int = 10


@dataclass
class ModelState:
    """State tracking for individual model in multi-model training"""
    model_id: str
    model_type: str
    model_name: str
    
    # Performance metrics
    total_reward: float = 0.0
    episode_count: int = 0
    win_count: int = 0  # For competitive scenarios
    cooperation_score: float = 0.5
    
    # Weight update tracking
    last_weight_update: float = 0.0
    update_count: int = 0
    gradient_norm: float = 0.0
    
    # Resource usage
    gpu_id: int = 0
    memory_usage: float = 0.0
    compute_utilization: float = 0.0
    
    # LoRA state
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_performance: float = 0.5


@dataclass
class TrainingCheckpoint:
    """Training checkpoint for multi-model system"""
    checkpoint_id: str
    timestamp: str
    episode: int
    
    # Model states
    model_states: Dict[str, ModelState]
    
    # Training metrics
    global_metrics: Dict[str, Any]
    
    # System state
    config: MultiModelConfig
    
    # File paths
    model_weights_dir: str
    lora_weights_dir: str
    
    def save_to_disk(self, base_path: str):
        """Save checkpoint to disk"""
        checkpoint_path = Path(base_path) / self.checkpoint_id
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        metadata = {
            "checkpoint_id": self.checkpoint_id,
            "timestamp": self.timestamp,
            "episode": self.episode,
            "model_states": {k: v.__dict__ for k, v in self.model_states.items()},
            "global_metrics": self.global_metrics,
            "config": self.config.__dict__
        }
        
        with open(checkpoint_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")


class MultiModelTrainer:
    """Core multi-model RL trainer with weight updates and VERL/AReaL integration"""
    
    def __init__(self, config: MultiModelConfig):
        self.config = config
        self.model_states = {}
        self.training_history = []
        self.checkpoints = []
        
        # Initialize components
        self._setup_models()
        self._setup_rl_system()
        self._setup_environment()
        self._setup_scheduler()
        self._setup_monitoring()
        self._setup_verl_areal()
        
        # Training state
        self.current_episode = 0
        self.is_training = False
        self.training_start_time = 0.0
        
        logger.info(f"Multi-model trainer initialized with {config.num_models} models")
    
    def _setup_models(self):
        """Setup multiple LLM models"""
        self.llm_managers = {}
        
        for i in range(self.config.num_models):
            model_type = self.config.model_types[i % len(self.config.model_types)]
            model_name = self.config.model_names.get(model_type, "Qwen/Qwen2.5-14B-Instruct")
            model_id = f"model_{i}_{model_type}"
            
            # Create appropriate manager based on type
            if model_type == "qwen3":
                manager = create_qwen3_manager(model_name)
            elif model_type == "openai":
                manager = create_openai_manager(model_name)
            elif model_type == "claude":
                manager = create_claude_manager(model_name)
            else:
                manager = create_qwen3_manager(model_name)  # Default to Qwen3
            
            self.llm_managers[model_id] = manager
            
            # Initialize model state
            self.model_states[model_id] = ModelState(
                model_id=model_id,
                model_type=model_type,
                model_name=model_name,
                gpu_id=i % self.config.num_gpus
            )
            
            logger.info(f"Initialized model {model_id}: {model_name}")
    
    def _setup_rl_system(self):
        """Setup multi-agent RL system"""
        from .rl_framework import create_multi_agent_system, CooperationFactor, CompetenceFactor, CooperationType, CompetenceType
        
        # Create cooperation factors for each model
        cooperation_configs = []
        competence_configs = []
        
        for i in range(self.config.num_models):
            # Vary cooperation strategies
            if i < self.config.num_models // 2:
                coop_type = CooperationType.TEAM_BASED
                coop_strength = self.config.cooperation_strength
            else:
                coop_type = CooperationType.SHARED_REWARDS
                coop_strength = self.config.cooperation_strength * 0.8
            
            cooperation_configs.append(CooperationFactor(
                cooperation_type=coop_type,
                cooperation_strength=coop_strength,
                team_size=2,
                shared_reward_ratio=0.7
            ))
            
            # Vary competence types
            comp_type = CompetenceType.ADAPTIVE if i % 2 == 0 else CompetenceType.SPECIALIZED
            competence_configs.append(CompetenceFactor(
                competence_type=comp_type,
                base_capability=0.5 + (i * 0.1),
                learning_rate=self.config.learning_rate,
                adaptation_speed=0.1
            ))
        
        self.rl_system = create_multi_agent_system(
            num_agents=self.config.num_models,
            enable_cooperation=True
        )
        
        # Create individual RL trainers for each model
        self.rl_trainers = {}
        for model_id, manager in self.llm_managers.items():
            trainer = create_ppo_trainer(manager, self.config.learning_rate)
            self.rl_trainers[model_id] = trainer
        
        logger.info(f"RL system initialized with {self.config.num_models} agents")
    
    def _setup_environment(self):
        """Setup training environment"""
        self.environment = create_multi_model_coop_compete_env(
            num_models=self.config.num_models,
            cooperation_level=self.config.cooperation_strength
        )
        logger.info("Multi-model training environment created")
    
    def _setup_scheduler(self):
        """Setup unified scheduler for resource management"""
        self.scheduler = create_unified_scheduler(
            base_port=self.config.base_port,
            num_gpus=self.config.num_gpus
        )
        
        # Register all models with scheduler
        for model_id, state in self.model_states.items():
            from .scheduler import ModelRole
            self.scheduler.register_model(model_id, state.gpu_id, ModelRole.GENERALIST)
        
        logger.info("Scheduler initialized with resource management")
    
    def _setup_monitoring(self):
        """Setup monitoring system"""
        if self.config.enable_monitoring:
            self.monitor = create_unified_monitor()
            self.monitor.start()
            logger.info("Monitoring system started")
        else:
            self.monitor = None
    
    def _setup_verl_areal(self):
        """Setup VERL and AReaL integration"""
        if self.config.enable_verl and self.config.enable_areal:
            from .cache_optimizer import create_areal_verl_bridge
            self.verl_areal_bridge = create_areal_verl_bridge()
            logger.info("VERL/AReaL integration initialized")
        else:
            self.verl_areal_bridge = None
    
    async def train_multi_model_episode(self, episode_num: int) -> Dict[str, Any]:
        """Train one episode with all models"""
        episode_start = time.time()
        episode_results = {}
        
        # Reset environment
        env_state = self.environment.reset()
        
        # Collect actions from all models
        model_actions = {}
        model_observations = {}
        
        for model_id in self.llm_managers.keys():
            # Get observation for this model
            obs = self.environment.get_observation(model_id)
            model_observations[model_id] = obs
            
            # Generate action using RL system
            action, log_prob, value = self.rl_system.step(model_id, obs)
            model_actions[model_id] = {
                "action": action,
                "log_prob": log_prob,
                "value": value
            }
        
        # Execute environment step
        actions_for_env = {k: 1 if v["action"] == "cooperate" else 0 for k, v in model_actions.items()}
        next_state, rewards, done, info = self.environment.step(actions_for_env)
        
        # Update each model with RL
        model_losses = {}
        weight_updates = {}
        
        for model_id, reward in rewards.items():
            if model_id in self.rl_trainers:
                # Store experience
                action_data = model_actions[model_id]
                self.rl_trainers[model_id].add_experience(
                    state=model_observations[model_id],
                    action=action_data["action"],
                    reward=reward,
                    done=done
                )
                
                # Update policy if enough data
                update_result = self.rl_trainers[model_id].update_policy()
                model_losses[model_id] = update_result
                
                # Extract weight gradients for multi-model coordination
                if update_result.get("status") == "updated":
                    weight_updates[model_id] = self._extract_weight_gradients(model_id, update_result)
                
                # Update model state
                self.model_states[model_id].total_reward += reward
                self.model_states[model_id].episode_count += 1
                self.model_states[model_id].last_weight_update = time.time()
                
                if reward > 1.0:  # Win condition
                    self.model_states[model_id].win_count += 1
        
        # Apply multi-model weight coordination
        if weight_updates:
            coordination_result = await self._coordinate_weight_updates(weight_updates, episode_num)
            episode_results["weight_coordination"] = coordination_result
        
        # VERL/AReaL optimization
        if self.verl_areal_bridge:
            prompts = [f"Episode {episode_num} action for {mid}" for mid in self.llm_managers.keys()]
            verl_result = await self.verl_areal_bridge.integrated_training_loop(prompts, num_steps=1)
            episode_results["verl_areal"] = verl_result
        
        episode_time = time.time() - episode_start
        
        episode_results.update({
            "episode": episode_num,
            "episode_time": episode_time,
            "model_actions": model_actions,
            "rewards": rewards,
            "model_losses": model_losses,
            "environment_info": info,
            "total_reward": sum(rewards.values()),
            "cooperation_ratio": sum(1 for action in actions_for_env.values() if action == 1) / len(actions_for_env)
        })
        
        return episode_results
    
    def _extract_weight_gradients(self, model_id: str, update_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract weight gradients from RL update for multi-model coordination"""
        # Simplified gradient extraction - in practice this would extract actual gradients
        return {
            "policy_gradient_norm": abs(update_result.get("losses", {}).get("policy_loss", 0.0)),
            "value_gradient_norm": abs(update_result.get("losses", {}).get("value_loss", 0.0)),
            "update_magnitude": update_result.get("losses", {}).get("total_loss", 0.0),
            "model_performance": self.model_states[model_id].total_reward / max(1, self.model_states[model_id].episode_count)
        }
    
    async def _coordinate_weight_updates(self, weight_updates: Dict[str, Dict[str, Any]], episode: int) -> Dict[str, Any]:
        """Coordinate weight updates across models"""
        coordination_result = {
            "strategy": self.config.weight_update_strategy.value,
            "episode": episode,
            "models_updated": [],
            "synchronization_loss": 0.0
        }
        
        if self.config.weight_update_strategy == WeightUpdateStrategy.SYNCHRONIZED:
            # All models update together with averaged gradients
            avg_gradients = self._average_gradients(weight_updates)
            
            for model_id in self.llm_managers.keys():
                await self._apply_coordinated_update(model_id, avg_gradients)
                coordination_result["models_updated"].append(model_id)
                
        elif self.config.weight_update_strategy == WeightUpdateStrategy.SELECTIVE:
            # Only top performing models update
            performance_scores = {
                mid: self.model_states[mid].total_reward / max(1, self.model_states[mid].episode_count)
                for mid in weight_updates.keys()
            }
            
            # Select top 50% performers
            top_models = sorted(performance_scores.items(), key=lambda x: x[1], reverse=True)
            top_models = [mid for mid, _ in top_models[:len(top_models)//2]]
            
            for model_id in top_models:
                if model_id in weight_updates:
                    await self._apply_coordinated_update(model_id, weight_updates[model_id])
                    coordination_result["models_updated"].append(model_id)
                    
        elif self.config.weight_update_strategy == WeightUpdateStrategy.FEDERATED:
            # Federated learning style coordination
            global_gradients = self._federated_averaging(weight_updates)
            
            for model_id in self.llm_managers.keys():
                local_gradients = weight_updates.get(model_id, {})
                combined_gradients = self._combine_gradients(local_gradients, global_gradients)
                await self._apply_coordinated_update(model_id, combined_gradients)
                coordination_result["models_updated"].append(model_id)
        
        else:  # ASYNCHRONOUS
            # Models update independently
            for model_id, gradients in weight_updates.items():
                await self._apply_coordinated_update(model_id, gradients)
                coordination_result["models_updated"].append(model_id)
        
        return coordination_result
    
    def _average_gradients(self, weight_updates: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Average gradients across models"""
        if not weight_updates:
            return {}
        
        avg_gradients = {}
        for key in ["policy_gradient_norm", "value_gradient_norm", "update_magnitude"]:
            values = [update.get(key, 0.0) for update in weight_updates.values()]
            avg_gradients[key] = sum(values) / len(values) if values else 0.0
        
        return avg_gradients
    
    def _federated_averaging(self, weight_updates: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Federated averaging of model updates"""
        # Weight by model performance
        total_performance = sum(
            self.model_states[mid].total_reward / max(1, self.model_states[mid].episode_count)
            for mid in weight_updates.keys()
        )
        
        weighted_gradients = {}
        for key in ["policy_gradient_norm", "value_gradient_norm", "update_magnitude"]:
            weighted_sum = 0.0
            for model_id, gradients in weight_updates.items():
                model_performance = self.model_states[model_id].total_reward / max(1, self.model_states[model_id].episode_count)
                weight = model_performance / max(total_performance, 1e-8)
                weighted_sum += weight * gradients.get(key, 0.0)
            weighted_gradients[key] = weighted_sum
        
        return weighted_gradients
    
    def _combine_gradients(self, local_gradients: Dict[str, Any], global_gradients: Dict[str, Any]) -> Dict[str, Any]:
        """Combine local and global gradients"""
        combined = {}
        alpha = 0.7  # Local weight
        
        for key in ["policy_gradient_norm", "value_gradient_norm", "update_magnitude"]:
            local_val = local_gradients.get(key, 0.0)
            global_val = global_gradients.get(key, 0.0)
            combined[key] = alpha * local_val + (1 - alpha) * global_val
        
        return combined
    
    async def _apply_coordinated_update(self, model_id: str, gradients: Dict[str, Any]):
        """Apply coordinated weight update to specific model"""
        if model_id not in self.llm_managers:
            return
        
        manager = self.llm_managers[model_id]
        
        # Convert gradients to format expected by LLM manager
        gradient_dict = {
            "policy_weights": gradients.get("policy_gradient_norm", 0.0),
            "value_weights": gradients.get("value_gradient_norm", 0.0)
        }
        
        # Apply update
        update_result = manager.update_shared_parameters(gradient_dict, self.config.learning_rate)
        
        # Update model state
        self.model_states[model_id].update_count += 1
        self.model_states[model_id].gradient_norm = gradients.get("update_magnitude", 0.0)
        
        logger.debug(f"Applied coordinated update to {model_id}")
    
    async def train(self) -> Dict[str, Any]:
        """Main training loop for multi-model system"""
        logger.info(f"Starting multi-model training for {self.config.max_episodes} episodes")
        
        self.is_training = True
        self.training_start_time = time.time()
        training_metrics = {
            "episodes": [],
            "total_rewards": [],
            "cooperation_ratios": [],
            "weight_update_counts": [],
            "model_performances": defaultdict(list)
        }
        
        try:
            for episode in range(self.config.max_episodes):
                self.current_episode = episode
                
                # Train episode
                episode_result = await self.train_multi_model_episode(episode)
                
                # Record metrics
                training_metrics["episodes"].append(episode)
                training_metrics["total_rewards"].append(episode_result["total_reward"])
                training_metrics["cooperation_ratios"].append(episode_result["cooperation_ratio"])
                
                # Record individual model performances
                for model_id, reward in episode_result["rewards"].items():
                    training_metrics["model_performances"][model_id].append(reward)
                
                # Update monitoring
                if self.monitor and episode % self.config.log_metrics_interval == 0:
                    await self._update_monitoring(episode_result)
                
                # Save checkpoint
                if episode % self.config.save_interval == 0:
                    await self._save_checkpoint(episode)
                
                # Log progress
                if episode % 50 == 0:
                    avg_reward = sum(training_metrics["total_rewards"][-50:]) / min(50, len(training_metrics["total_rewards"]))
                    avg_coop = sum(training_metrics["cooperation_ratios"][-50:]) / min(50, len(training_metrics["cooperation_ratios"]))
                    logger.info(f"Episode {episode}: avg_reward={avg_reward:.3f}, cooperation={avg_coop:.3f}")
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.01)
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            self.is_training = False
            
            # Final checkpoint
            await self._save_checkpoint(self.current_episode, final=True)
        
        training_time = time.time() - self.training_start_time
        
        final_results = {
            "training_completed": True,
            "total_episodes": self.current_episode,
            "training_time": training_time,
            "training_metrics": training_metrics,
            "final_model_states": {k: v.__dict__ for k, v in self.model_states.items()},
            "checkpoints_saved": len(self.checkpoints)
        }
        
        logger.info(f"Multi-model training completed: {self.current_episode} episodes in {training_time:.2f}s")
        return final_results
    
    async def _update_monitoring(self, episode_result: Dict[str, Any]):
        """Update monitoring with episode results"""
        if not self.monitor:
            return
        
        from .monitoring import create_social_network_metrics
        
        # Create metrics from episode results
        metrics = create_social_network_metrics(
            total_users=self.config.num_models,
            active_users=len([m for m in self.model_states.values() if m.episode_count > 0]),
            engagement_rate=episode_result["cooperation_ratio"],
            response_time_avg=episode_result["episode_time"],
            avg_influence_score=episode_result["total_reward"] / self.config.num_models
        )
        
        self.monitor.update_metrics(metrics)
    
    async def _save_checkpoint(self, episode: int, final: bool = False):
        """Save training checkpoint"""
        checkpoint_id = f"multimodel_ep_{episode}_{int(time.time())}"
        if final:
            checkpoint_id = f"final_{checkpoint_id}"
        
        # Create checkpoint directory
        checkpoint_dir = Path(self.config.checkpoint_dir) / checkpoint_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        model_weights_dir = checkpoint_dir / "model_weights"
        model_weights_dir.mkdir(exist_ok=True)
        
        for model_id, manager in self.llm_managers.items():
            model_params = manager.get_global_stats()
            model_file = model_weights_dir / f"{model_id}.json"
            
            with open(model_file, 'w') as f:
                json.dump(model_params, f, indent=2)
        
        # Save LoRA weights if available
        lora_weights_dir = checkpoint_dir / "lora_weights"
        lora_weights_dir.mkdir(exist_ok=True)
        
        # Create checkpoint object
        checkpoint = TrainingCheckpoint(
            checkpoint_id=checkpoint_id,
            timestamp=datetime.now().isoformat(),
            episode=episode,
            model_states=self.model_states.copy(),
            global_metrics={
                "total_episodes": episode,
                "training_time": time.time() - self.training_start_time,
                "avg_cooperation": sum(
                    state.cooperation_score for state in self.model_states.values()
                ) / len(self.model_states)
            },
            config=self.config,
            model_weights_dir=str(model_weights_dir),
            lora_weights_dir=str(lora_weights_dir)
        )
        
        # Save checkpoint metadata
        checkpoint.save_to_disk(self.config.checkpoint_dir)
        
        # Add to checkpoint list
        self.checkpoints.append(checkpoint)
        
        # Cleanup old checkpoints
        if len(self.checkpoints) > self.config.max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            self._cleanup_checkpoint(old_checkpoint)
        
        logger.info(f"Checkpoint saved: {checkpoint_id}")
    
    def _cleanup_checkpoint(self, checkpoint: TrainingCheckpoint):
        """Clean up old checkpoint files"""
        try:
            checkpoint_path = Path(self.config.checkpoint_dir) / checkpoint.checkpoint_id
            if checkpoint_path.exists():
                import shutil
                shutil.rmtree(checkpoint_path)
                logger.debug(f"Cleaned up old checkpoint: {checkpoint.checkpoint_id}")
        except Exception as e:
            logger.warning(f"Failed to cleanup checkpoint: {e}")
    
    def load_checkpoint(self, checkpoint_id: str) -> bool:
        """Load training checkpoint"""
        try:
            checkpoint_path = Path(self.config.checkpoint_dir) / checkpoint_id
            metadata_file = checkpoint_path / "metadata.json"
            
            if not metadata_file.exists():
                logger.error(f"Checkpoint metadata not found: {metadata_file}")
                return False
            
            # Load metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Restore model states
            for model_id, state_dict in metadata["model_states"].items():
                if model_id in self.model_states:
                    for key, value in state_dict.items():
                        setattr(self.model_states[model_id], key, value)
            
            # Restore training state
            self.current_episode = metadata["episode"]
            
            logger.info(f"Checkpoint loaded: {checkpoint_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status"""
        return {
            "is_training": self.is_training,
            "current_episode": self.current_episode,
            "max_episodes": self.config.max_episodes,
            "progress": self.current_episode / self.config.max_episodes if self.config.max_episodes > 0 else 0.0,
            "training_time": time.time() - self.training_start_time if self.training_start_time > 0 else 0.0,
            "model_states": {k: v.__dict__ for k, v in self.model_states.items()},
            "checkpoints_count": len(self.checkpoints),
            "config": self.config.__dict__
        }
    
    def get_model_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all models"""
        summary = {}
        
        for model_id, state in self.model_states.items():
            avg_reward = state.total_reward / max(1, state.episode_count)
            win_rate = state.win_count / max(1, state.episode_count)
            
            summary[model_id] = {
                "model_type": state.model_type,
                "model_name": state.model_name,
                "avg_reward": avg_reward,
                "win_rate": win_rate,
                "total_episodes": state.episode_count,
                "update_count": state.update_count,
                "cooperation_score": state.cooperation_score,
                "gpu_id": state.gpu_id
            }
        
        return summary
    
    async def shutdown(self):
        """Shutdown multi-model trainer"""
        logger.info("Shutting down multi-model trainer...")
        
        # Stop monitoring
        if self.monitor:
            self.monitor.stop()
        
        # Save final checkpoint if training
        if self.is_training:
            await self._save_checkpoint(self.current_episode, final=True)
        
        # Shutdown scheduler
        if hasattr(self, 'scheduler'):
            await self.scheduler.shutdown()
        
        logger.info("Multi-model trainer shutdown complete")


# Factory functions for quick setup
def create_multimodel_trainer(num_models: int = 4, training_mode: TrainingMode = TrainingMode.MIXED,
                             model_types: Optional[List[str]] = None) -> MultiModelTrainer:
    """Create multi-model trainer with specified configuration"""
    
    if model_types is None:
        model_types = ["qwen3", "openai", "claude", "llama3"]
    
    config = MultiModelConfig(
        num_models=num_models,
        model_types=model_types,
        training_mode=training_mode,
        weight_update_strategy=WeightUpdateStrategy.ASYNCHRONOUS,
        max_episodes=1000,
        learning_rate=3e-4,
        enable_verl=True,
        enable_areal=True,
        enable_monitoring=True
    )
    
    return MultiModelTrainer(config)


def create_cooperative_multimodel_trainer(num_models: int = 4) -> MultiModelTrainer:
    """Create cooperative multi-model trainer"""
    return create_multimodel_trainer(
        num_models=num_models,
        training_mode=TrainingMode.COOPERATIVE,
        model_types=["qwen3"] * num_models  # Use same model type for fair comparison
    )


def create_competitive_multimodel_trainer(num_models: int = 4) -> MultiModelTrainer:
    """Create competitive multi-model trainer"""
    return create_multimodel_trainer(
        num_models=num_models,
        training_mode=TrainingMode.COMPETITIVE,
        model_types=["qwen3", "openai", "claude", "llama3"]  # Different models compete
    )


async def quick_start_multimodel_training(num_models: int = 4, max_episodes: int = 100) -> Dict[str, Any]:
    """Quick start multi-model training session"""
    
    # Create trainer
    trainer = create_multimodel_trainer(num_models=num_models)
    trainer.config.max_episodes = max_episodes
    
    try:
        # Start training
        results = await trainer.train()
        
        # Get final performance
        performance = trainer.get_model_performance_summary()
        
        return {
            "training_results": results,
            "model_performance": performance,
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"Quick start training failed: {e}")
        return {
            "status": "failed",
            "error": str(e)
        }
    finally:
        await trainer.shutdown()


# Checkpoint utilities
def list_available_checkpoints(checkpoint_dir: str = "./checkpoints/multimodel") -> List[str]:
    """List available checkpoints"""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return []
    
    checkpoints = []
    for item in checkpoint_path.iterdir():
        if item.is_dir() and (item / "metadata.json").exists():
            checkpoints.append(item.name)
    
    return sorted(checkpoints, reverse=True)  # Most recent first


def load_checkpoint_metadata(checkpoint_id: str, checkpoint_dir: str = "./checkpoints/multimodel") -> Optional[Dict[str, Any]]:
    """Load checkpoint metadata"""
    try:
        metadata_file = Path(checkpoint_dir) / checkpoint_id / "metadata.json"
        with open(metadata_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load checkpoint metadata: {e}")
        return None
