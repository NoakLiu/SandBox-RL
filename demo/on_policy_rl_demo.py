#!/usr/bin/env python3
"""
On-Policy RL Framework Demo

This script demonstrates the usage of the on-policy RL framework with:
- Cooperation Factor: Controls collaborative behavior between agents
- Competence Factor: Controls individual agent capability and learning
- Multi-LoRA support for single vLLM instance
- LlamaFactory integration for LoRA parameter initialization
"""

import numpy as np
import time
import logging
import json
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns

# Sandbox-RL Core imports
try:
    from sandbox_rl.core.on_policy_rl_framework import (
        CooperationType,
        CompetenceType,
        CooperationFactor,
        CompetenceFactor,
        create_on_policy_rl_system,
        initialize_vllm_lora_adapters,
        MultiLoRAOnPolicyRL
    )
    HAS_SANDGRAPH = True
    print("✅ Sandbox-RL on-policy RL framework imported successfully")
except ImportError as e:
    HAS_SANDGRAPH = False
    print(f"❌ Sandbox-RL on-policy RL framework not available: {e}")
    print("Will use mock implementations")

# Optional imports
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockEnvironment:
    """Mock environment for demonstration"""
    
    def __init__(self, state_dim: int = 64, action_dim: int = 10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.current_state = np.random.randn(state_dim)
        self.step_count = 0
        self.max_steps = 100
        
    def reset(self) -> np.ndarray:
        """Reset environment"""
        self.current_state = np.random.randn(self.state_dim)
        self.step_count = 0
        return self.current_state
    
    def step(self, action: int) -> tuple:
        """Take a step in the environment"""
        # Simulate state transition
        noise = np.random.randn(self.state_dim) * 0.1
        self.current_state = self.current_state + noise
        
        # Simulate reward based on action and state
        reward = np.sin(self.step_count * 0.1) + np.random.normal(0, 0.1)
        reward += 0.1 * (action / self.action_dim)  # Action bonus
        
        self.step_count += 1
        done = self.step_count >= self.max_steps
        
        return self.current_state, reward, done, {}

def demonstrate_cooperation_factors():
    """Demonstrate different cooperation factors"""
    print("\n🔗 Cooperation Factors Demonstration")
    print("=" * 50)
    
    # Create different cooperation configurations
    cooperation_configs = [
        ("No Cooperation", CooperationFactor(
            cooperation_type=CooperationType.NONE,
            cooperation_strength=0.0
        )),
        ("Team-Based Cooperation", CooperationFactor(
            cooperation_type=CooperationType.TEAM_BASED,
            cooperation_strength=0.3,
            team_size=4,
            shared_reward_ratio=0.6
        )),
        ("Shared Rewards", CooperationFactor(
            cooperation_type=CooperationType.SHARED_REWARDS,
            cooperation_strength=0.2,
            shared_reward_ratio=0.8
        )),
        ("Knowledge Transfer", CooperationFactor(
            cooperation_type=CooperationType.KNOWLEDGE_TRANSFER,
            cooperation_strength=0.4,
            knowledge_transfer_rate=0.15
        ))
    ]
    
    for i, (description, config) in enumerate(cooperation_configs):
        print(f"\n{i+1}. {description}")
        print(f"   - Cooperation Type: {config.cooperation_type.value}")
        print(f"   - Cooperation Strength: {config.cooperation_strength}")
        print(f"   - Team Size: {config.team_size}")
        print(f"   - Shared Reward Ratio: {config.shared_reward_ratio}")
        print(f"   - Knowledge Transfer Rate: {config.knowledge_transfer_rate}")

def demonstrate_competence_factors():
    """Demonstrate different competence factors"""
    print("\n🎯 Competence Factors Demonstration")
    print("=" * 50)
    
    # Create different competence configurations
    competence_configs = [
        ("Novice Agent", CompetenceFactor(
            competence_type=CompetenceType.NOVICE,
            base_capability=0.3,
            learning_rate=0.01,
            adaptation_speed=0.05
        )),
        ("General Agent", CompetenceFactor(
            competence_type=CompetenceType.GENERAL,
            base_capability=0.5,
            learning_rate=0.02,
            adaptation_speed=0.1
        )),
        ("Specialized Agent", CompetenceFactor(
            competence_type=CompetenceType.SPECIALIZED,
            base_capability=0.6,
            learning_rate=0.03,
            specialization_level=0.4
        )),
        ("Adaptive Agent", CompetenceFactor(
            competence_type=CompetenceType.ADAPTIVE,
            base_capability=0.4,
            learning_rate=0.025,
            adaptation_speed=0.15
        )),
        ("Expert Agent", CompetenceFactor(
            competence_type=CompetenceType.EXPERT,
            base_capability=0.8,
            learning_rate=0.01,
            specialization_level=0.6
        ))
    ]
    
    for i, (description, config) in enumerate(competence_configs):
        print(f"\n{i+1}. {description}")
        print(f"   - Competence Type: {config.competence_type.value}")
        print(f"   - Base Capability: {config.base_capability}")
        print(f"   - Learning Rate: {config.learning_rate}")
        print(f"   - Adaptation Speed: {config.adaptation_speed}")
        print(f"   - Specialization Level: {config.specialization_level}")

def run_on_policy_rl_training():
    """Run on-policy RL training demonstration"""
    print("\n🚀 On-Policy RL Training Demonstration")
    print("=" * 50)
    
    if not HAS_SANDGRAPH:
        print("❌ Sandbox-RL not available, skipping training demo")
        return
    
    # Create on-policy RL system
    num_adapters = 8
    state_dim = 64
    action_dim = 10
    
    print(f"Creating on-policy RL system with {num_adapters} adapters...")
    
    # Create system with team-based cooperation and adaptive competence
    rl_system = create_on_policy_rl_system(
        num_adapters=num_adapters,
        cooperation_type=CooperationType.TEAM_BASED,
        competence_type=CompetenceType.ADAPTIVE,
        device="cpu"
    )
    
    # Create mock environment
    env = MockEnvironment(state_dim=state_dim, action_dim=action_dim)
    
    # Training parameters
    num_episodes = 50
    max_steps_per_episode = 100
    update_frequency = 10
    
    # Training loop
    episode_rewards = []
    agent_stats_history = []
    
    print(f"Starting training for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0
        
        for step in range(max_steps_per_episode):
            # Randomly select an adapter for demonstration
            adapter_id = str(step % num_adapters)
            
            # Get action from RL system
            action, log_prob, value = rl_system.step(adapter_id, state)
            
            # Take action in environment
            next_state, reward, done, _ = env.step(action)
            
            # Update agent
            rl_system.update_agent(
                adapter_id=adapter_id,
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                log_prob=log_prob,
                value=value
            )
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
        # Update policies periodically
        if (episode + 1) % update_frequency == 0:
            rl_system.update_all_policies()
            print(f"Episode {episode + 1}: Reward = {episode_reward:.3f}")
        
        # Collect agent stats
        if episode % 5 == 0:
            agent_stats = rl_system.get_agent_stats()
            agent_stats_history.append({
                'episode': episode,
                'stats': agent_stats
            })
    
    print(f"Training completed! Average reward: {np.mean(episode_rewards):.3f}")
    
    # Plot training results
    plot_training_results(episode_rewards, agent_stats_history)
    
    return rl_system, episode_rewards, agent_stats_history

def plot_training_results(episode_rewards: List[float], agent_stats_history: List[Dict]):
    """Plot training results"""
    print("\n📊 Plotting Training Results")
    print("=" * 30)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Episode rewards
    axes[0, 0].plot(episode_rewards, alpha=0.7, color='blue')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Moving average rewards
    window_size = 10
    moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
    axes[0, 1].plot(moving_avg, alpha=0.7, color='red')
    axes[0, 1].set_title(f'Moving Average Rewards (window={window_size})')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Average Reward')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Agent capabilities over time
    if agent_stats_history:
        episodes = [stats['episode'] for stats in agent_stats_history]
        capabilities = []
        for stats in agent_stats_history:
            avg_capability = np.mean([agent['capability'] for agent in stats['stats'].values()])
            capabilities.append(avg_capability)
        
        axes[1, 0].plot(episodes, capabilities, alpha=0.7, color='green')
        axes[1, 0].set_title('Average Agent Capability')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Capability')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Experience counts
    if agent_stats_history:
        experience_counts = []
        for stats in agent_stats_history:
            avg_experience = np.mean([agent['experience_count'] for agent in stats['stats'].values()])
            experience_counts.append(avg_experience)
        
        axes[1, 1].plot(episodes, experience_counts, alpha=0.7, color='purple')
        axes[1, 1].set_title('Average Experience Count')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Experience Count')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('on_policy_rl_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Training results plotted and saved as 'on_policy_rl_training_results.png'")

def demonstrate_lora_initialization():
    """Demonstrate LoRA initialization with LlamaFactory"""
    print("\n🔧 LoRA Initialization with LlamaFactory")
    print("=" * 50)
    
    if not HAS_SANDGRAPH:
        print("❌ Sandbox-RL not available, skipping LoRA initialization demo")
        return
    
    # Example LlamaFactory configuration
    llama_factory_config_path = "/path/to/llama_factory_config.json"
    pretrained_model_name = "meta-llama/Llama-2-7b-hf"
    output_dir = "./lora_adapters"
    num_adapters = 8
    
    print(f"Initializing {num_adapters} LoRA adapters...")
    print(f"LlamaFactory config: {llama_factory_config_path}")
    print(f"Pretrained model: {pretrained_model_name}")
    print(f"Output directory: {output_dir}")
    
    try:
        # Initialize LoRA adapters
        lora_configs = initialize_vllm_lora_adapters(
            llama_factory_config_path=llama_factory_config_path,
            pretrained_model_name=pretrained_model_name,
            output_dir=output_dir,
            num_adapters=num_adapters
        )
        
        print(f"✅ Successfully initialized {len(lora_configs)} LoRA adapters")
        
        # Display configuration for first adapter
        first_adapter = list(lora_configs.values())[0]
        print(f"\nFirst adapter configuration:")
        print(f"  - Adapter ID: {first_adapter.adapter_id}")
        print(f"  - Rank: {first_adapter.rank}")
        print(f"  - Alpha: {first_adapter.alpha}")
        print(f"  - Dropout: {first_adapter.dropout}")
        print(f"  - Target Modules: {first_adapter.target_modules}")
        print(f"  - Task Type: {first_adapter.task_type}")
        
    except Exception as e:
        print(f"❌ LoRA initialization failed: {e}")
        print("This is expected in demo mode without actual LlamaFactory setup")

def main():
    """Main demonstration function"""
    print("🚀 On-Policy RL Framework Demo")
    print("=" * 60)
    print("This demo showcases the on-policy RL framework with:")
    print("- Cooperation factors for multi-agent collaboration")
    print("- Competence factors for individual agent learning")
    print("- Multi-LoRA support for single vLLM instance")
    print("- LlamaFactory integration for LoRA parameter initialization")
    
    # Demonstrate cooperation factors
    demonstrate_cooperation_factors()
    
    # Demonstrate competence factors
    demonstrate_competence_factors()
    
    # Demonstrate LoRA initialization
    demonstrate_lora_initialization()
    
    # Run on-policy RL training
    if HAS_SANDGRAPH and HAS_TORCH:
        rl_system, episode_rewards, agent_stats_history = run_on_policy_rl_training()
        
        # Save results
        results = {
            'episode_rewards': episode_rewards,
            'final_agent_stats': rl_system.get_agent_stats() if rl_system else {},
            'training_config': {
                'num_adapters': 8,
                'cooperation_type': 'TEAM_BASED',
                'competence_type': 'ADAPTIVE'
            }
        }
        
        with open('on_policy_rl_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print("\n✅ Demo completed successfully!")
        print("📁 Results saved to:")
        print("  - on_policy_rl_training_results.png")
        print("  - on_policy_rl_results.json")
    else:
        print("\n⚠️  Training demo skipped due to missing dependencies")
        print("   Install torch and ensure Sandbox-RL is available for full demo")

if __name__ == "__main__":
    main()
