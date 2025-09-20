#!/usr/bin/env python3
"""
Cooperative Multi-Model Training Example
========================================

Example showing cooperative training where models help each other learn
through knowledge sharing and coordinated weight updates.
"""

import asyncio
import logging
from core_srl import (
    create_cooperative_multimodel_trainer,
    MultiModelConfig,
    TrainingMode,
    WeightUpdateStrategy
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def cooperative_training_example():
    """Train models to cooperate and share knowledge"""
    
    print(" Starting cooperative multi-model training...")
    print(" Configuration:")
    print("   - Models: 4 (same type for fair cooperation)")
    print("   - Mode: Cooperative")
    print("   - Strategy: Synchronized weight updates")
    
    # Create cooperative trainer
    trainer = create_cooperative_multimodel_trainer(num_models=4)
    
    # Configure for maximum cooperation
    trainer.config.cooperation_strength = 0.9  # Very high cooperation
    trainer.config.competition_intensity = 0.1  # Minimal competition
    trainer.config.weight_update_strategy = WeightUpdateStrategy.SYNCHRONIZED  # All update together
    trainer.config.max_episodes = 200
    trainer.config.checkpoint_dir = "./checkpoints/cooperative"
    
    # Use same model type for fair cooperation
    trainer.config.model_types = ["qwen3"] * 4
    trainer.config.model_names = {
        "qwen3": "Qwen/Qwen2.5-14B-Instruct"
    }
    
    try:
        # Start cooperative training
        results = await trainer.train()
        
        print("\n✅ Cooperative training completed!")
        
        # Analyze cooperation effectiveness
        performance = trainer.get_model_performance_summary()
        
        # Calculate cooperation metrics
        rewards = [stats['avg_reward'] for stats in performance.values()]
        cooperation_scores = [stats['cooperation_score'] for stats in performance.values()]
        
        mean_reward = sum(rewards) / len(rewards)
        reward_variance = sum((r - mean_reward)**2 for r in rewards) / len(rewards)
        mean_cooperation = sum(cooperation_scores) / len(cooperation_scores)
        
        print(f"\n Cooperation Analysis:")
        print(f"   Average Reward: {mean_reward:.3f}")
        print(f"   Reward Variance: {reward_variance:.4f} (lower = better cooperation)")
        print(f"   Cooperation Score: {mean_cooperation:.3f}")
        
        # Check if cooperation was successful
        if reward_variance < 0.01:
            print("   ✅ Excellent cooperation: Very similar performance across models")
        elif reward_variance < 0.05:
            print("   ✅ Good cooperation: Models learned together effectively")
        else:
            print("   ⚠️ Limited cooperation: Models diverged despite cooperation settings")
        
        # Show individual model contributions
        print(f"\n Individual Model Performance:")
        for model_id, stats in performance.items():
            contribution = (stats['avg_reward'] - mean_reward) / max(mean_reward, 0.001)
            print(f"   {model_id}:")
            print(f"     Reward: {stats['avg_reward']:.3f}")
            print(f"     Contribution: {contribution:+.1%}")
            print(f"     Cooperation Score: {stats['cooperation_score']:.3f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Cooperative training failed: {e}")
        return None
    
    finally:
        await trainer.shutdown()


async def knowledge_sharing_analysis(trainer):
    """Analyze knowledge sharing between models"""
    
    print("\n Knowledge Sharing Analysis:")
    
    # Get training status
    status = trainer.get_training_status()
    model_states = status['model_states']
    
    # Analyze weight update patterns
    update_counts = [state['update_count'] for state in model_states.values()]
    update_variance = sum((u - sum(update_counts)/len(update_counts))**2 for u in update_counts) / len(update_counts)
    
    print(f"   Update Synchronization: {1.0/(1.0 + update_variance):.3f}")
    
    # Analyze gradient norms (indicator of learning activity)
    gradient_norms = [state.get('gradient_norm', 0.0) for state in model_states.values()]
    avg_gradient_norm = sum(gradient_norms) / len(gradient_norms)
    
    print(f"   Average Gradient Norm: {avg_gradient_norm:.4f}")
    
    # Check for knowledge convergence
    if update_variance < 1.0:
        print("   ✅ Models are learning in sync (good knowledge sharing)")
    else:
        print("   ⚠️ Models are learning at different rates")


async def team_formation_example():
    """Example showing team-based cooperative training"""
    
    print("\n Team-Based Cooperative Training")
    print("=" * 40)
    
    # Configure team-based training
    config = MultiModelConfig(
        num_models=6,
        model_types=["qwen3"] * 6,  # Same type for fair teams
        training_mode=TrainingMode.COOPERATIVE,
        weight_update_strategy=WeightUpdateStrategy.FEDERATED,
        cooperation_strength=0.8,
        max_episodes=150
    )
    
    from core_srl import MultiModelTrainer
    trainer = MultiModelTrainer(config)
    
    try:
        print(" Starting team-based training...")
        print("   Team A: Models 0, 1, 2")
        print("   Team B: Models 3, 4, 5")
        
        results = await trainer.train()
        
        # Analyze team performance
        performance = trainer.get_model_performance_summary()
        
        # Split into teams
        team_a_performance = {k: v for k, v in performance.items() if int(k.split('_')[1]) < 3}
        team_b_performance = {k: v for k, v in performance.items() if int(k.split('_')[1]) >= 3}
        
        team_a_avg = sum(p['avg_reward'] for p in team_a_performance.values()) / len(team_a_performance)
        team_b_avg = sum(p['avg_reward'] for p in team_b_performance.values()) / len(team_b_performance)
        
        print(f"\n Team Results:")
        print(f"   Team A Average: {team_a_avg:.3f}")
        print(f"   Team B Average: {team_b_avg:.3f}")
        
        if abs(team_a_avg - team_b_avg) < 0.05:
            print("    Balanced team performance (good cooperation)")
        else:
            winner = "Team A" if team_a_avg > team_b_avg else "Team B"
            print(f"    {winner} performed better")
        
        return results
        
    finally:
        await trainer.shutdown()


async def cooperative_knowledge_transfer():
    """Example of cooperative knowledge transfer between specialized models"""
    
    print("\n Cooperative Knowledge Transfer")
    print("=" * 40)
    
    # Use specialized models for knowledge transfer
    config = MultiModelConfig(
        num_models=3,
        model_types=["qwen_coder", "qwen_math", "qwen3"],
        model_names={
            "qwen_coder": "Qwen/Qwen2.5-Coder-14B-Instruct",  # Code specialist
            "qwen_math": "Qwen/Qwen2.5-Math-14B-Instruct",    # Math specialist
            "qwen3": "Qwen/Qwen2.5-14B-Instruct"              # Generalist
        },
        training_mode=TrainingMode.COOPERATIVE,
        weight_update_strategy=WeightUpdateStrategy.FEDERATED,  # Knowledge sharing
        cooperation_strength=0.9,
        max_episodes=100
    )
    
    from core_srl import MultiModelTrainer
    trainer = MultiModelTrainer(config)
    
    try:
        print(" Starting knowledge transfer training...")
        print("   Coder → Math → Generalist knowledge flow")
        
        results = await trainer.train()
        
        # Analyze specialization retention vs knowledge transfer
        performance = trainer.get_model_performance_summary()
        
        print(f"\n📚 Knowledge Transfer Results:")
        for model_id, stats in performance.items():
            model_type = stats['model_type']
            specialization = "" if "coder" in model_type else "🔢" if "math" in model_type else "🌐"
            
            print(f"   {specialization} {model_id} ({model_type}):")
            print(f"      Performance: {stats['avg_reward']:.3f}")
            print(f"      Cooperation: {stats['cooperation_score']:.3f}")
        
        # Check knowledge transfer effectiveness
        coder_perf = next(p['avg_reward'] for k, p in performance.items() if 'coder' in p['model_type'])
        math_perf = next(p['avg_reward'] for k, p in performance.items() if 'math' in p['model_type'])
        general_perf = next(p['avg_reward'] for k, p in performance.items() if p['model_type'] == 'qwen3')
        
        transfer_effectiveness = general_perf / max((coder_perf + math_perf) / 2, 0.001)
        
        print(f"\n Knowledge Transfer Effectiveness: {transfer_effectiveness:.3f}")
        if transfer_effectiveness > 0.9:
            print("   ✅ Excellent knowledge transfer: Generalist learned from specialists")
        elif transfer_effectiveness > 0.7:
            print("   ✅ Good knowledge transfer: Some specialist knowledge transferred")
        else:
            print("   ⚠️ Limited knowledge transfer: Specialists remained isolated")
        
        return results
        
    finally:
        await trainer.shutdown()


if __name__ == "__main__":
    print("=" * 60)
    print(" Core SRL - Cooperative Multi-Model Training")
    print("=" * 60)
    
    # Run main cooperative training
    results = asyncio.run(cooperative_training_example())
    
    # Run team-based training
    asyncio.run(team_formation_example())
    
    # Run knowledge transfer example
    asyncio.run(cooperative_knowledge_transfer())
    
    print("\n" + "=" * 60)
    print(" Cooperative training examples completed!")
    print(" Key insights:")
    print("   - Synchronized updates improve cooperation")
    print("   - Same model types cooperate better than different types")
    print("   - Knowledge transfer works between specialized models")
    print("   - Team-based training balances individual and group performance")
    print("=" * 60)
