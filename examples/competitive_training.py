#!/usr/bin/env python3
"""
Competitive Multi-Model Training Example
========================================

Example showing competitive training where different model types compete
against each other for performance ranking and resource allocation.
"""

import asyncio
import logging
from core_srl import (
    create_competitive_multimodel_trainer,
    MultiModelConfig,
    TrainingMode,
    WeightUpdateStrategy,
    list_available_checkpoints
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def competitive_training_example():
    """Train different model types in competitive mode"""
    
    print("⚔️ Starting competitive multi-model training...")
    print("📋 Configuration:")
    print("   - Models: Qwen3 vs OpenAI vs Claude vs Llama3")
    print("   - Mode: Competitive")
    print("   - Strategy: Only top performers get weight updates")
    
    # Create competitive trainer with different model types
    trainer = create_competitive_multimodel_trainer(num_models=4)
    
    # Configure for intense competition
    trainer.config.model_types = ["qwen3", "openai", "claude", "llama3"]
    trainer.config.model_names = {
        "qwen3": "Qwen/Qwen2.5-14B-Instruct",
        "openai": "gpt-4o-mini",
        "claude": "claude-3-5-haiku-20241022", 
        "llama3": "meta-llama/Llama-3.1-8B-Instruct"
    }
    trainer.config.competition_intensity = 0.8  # High competition
    trainer.config.cooperation_strength = 0.2   # Low cooperation
    trainer.config.weight_update_strategy = WeightUpdateStrategy.SELECTIVE  # Only winners update
    trainer.config.max_episodes = 200
    trainer.config.checkpoint_dir = "./checkpoints/competitive"
    
    try:
        # Start competitive training
        results = await trainer.train()
        
        print("\n🏁 Competitive training completed!")
        
        # Analyze competition results
        performance = trainer.get_model_performance_summary()
        
        # Rank models by performance
        ranked_models = sorted(
            performance.items(),
            key=lambda x: x[1]['avg_reward'],
            reverse=True
        )
        
        print("\n🏆 Competition Rankings:")
        for rank, (model_id, stats) in enumerate(ranked_models, 1):
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
            print(f"   {medal} #{rank} {model_id} ({stats['model_type']}):")
            print(f"      Avg Reward: {stats['avg_reward']:.3f}")
            print(f"      Win Rate: {stats['win_rate']:.3f}")
            print(f"      Weight Updates: {stats['update_count']}")
        
        # Competition analysis
        rewards = [stats['avg_reward'] for stats in performance.values()]
        max_reward = max(rewards)
        min_reward = min(rewards)
        performance_gap = max_reward - min_reward
        
        print(f"\n📊 Competition Analysis:")
        print(f"   Performance Gap: {performance_gap:.3f}")
        print(f"   Winner Advantage: {(max_reward/min_reward - 1)*100:.1f}%")
        
        # Check for emergent specialization
        update_counts = [stats['update_count'] for stats in performance.values()]
        max_updates = max(update_counts)
        min_updates = min(update_counts)
        update_inequality = (max_updates - min_updates) / max(max_updates, 1)
        
        print(f"   Update Inequality: {update_inequality:.3f}")
        if update_inequality > 0.5:
            print("   🎯 Strong competitive dynamics detected!")
        
        return results
        
    except Exception as e:
        logger.error(f"Competitive training failed: {e}")
        return None
    
    finally:
        await trainer.shutdown()


async def analyze_competitive_dynamics(trainer):
    """Analyze competitive dynamics during training"""
    
    print("\n🔍 Analyzing competitive dynamics...")
    
    # Get training status
    status = trainer.get_training_status()
    model_states = status['model_states']
    
    # Calculate competition metrics
    competition_metrics = {}
    
    for model_id, state in model_states.items():
        # Resource utilization competition
        gpu_utilization = state.get('compute_utilization', 0.5)
        memory_usage = state.get('memory_usage', 0.5)
        
        # Performance competition  
        avg_reward = state['total_reward'] / max(1, state['episode_count'])
        win_rate = state['win_count'] / max(1, state['episode_count'])
        
        # Update frequency (indicator of learning activity)
        update_frequency = state['update_count'] / max(1, state['episode_count'])
        
        competition_metrics[model_id] = {
            'resource_competition': (gpu_utilization + memory_usage) / 2,
            'performance_competition': avg_reward,
            'learning_competition': update_frequency,
            'win_rate': win_rate
        }
    
    # Find most competitive model
    most_competitive = max(
        competition_metrics.items(),
        key=lambda x: sum(x[1].values())
    )
    
    print(f"   🏅 Most Competitive Model: {most_competitive[0]}")
    print(f"   📈 Competition Score: {sum(most_competitive[1].values()):.3f}")
    
    return competition_metrics


async def competitive_vs_cooperative_comparison():
    """Compare competitive vs cooperative training"""
    
    print("\n🆚 Competitive vs Cooperative Comparison")
    print("=" * 50)
    
    # Test competitive training
    print("⚔️ Testing competitive training...")
    competitive_trainer = create_competitive_multimodel_trainer(num_models=4)
    competitive_trainer.config.max_episodes = 100
    competitive_results = await competitive_trainer.train()
    await competitive_trainer.shutdown()
    
    # Test cooperative training  
    print("\n🤝 Testing cooperative training...")
    from core_srl import create_cooperative_multimodel_trainer
    cooperative_trainer = create_cooperative_multimodel_trainer(num_models=4)
    cooperative_trainer.config.max_episodes = 100
    cooperative_results = await cooperative_trainer.train()
    await cooperative_trainer.shutdown()
    
    # Compare results
    comp_performance = competitive_results['model_performance']
    coop_performance = cooperative_results['model_performance']
    
    comp_rewards = [p['avg_reward'] for p in comp_performance.values()]
    coop_rewards = [p['avg_reward'] for p in coop_performance.values()]
    
    comp_max = max(comp_rewards)
    comp_avg = sum(comp_rewards) / len(comp_rewards)
    coop_max = max(coop_rewards)
    coop_avg = sum(coop_rewards) / len(coop_rewards)
    
    print(f"\n📊 Comparison Results:")
    print(f"   Competitive - Max: {comp_max:.3f}, Avg: {comp_avg:.3f}")
    print(f"   Cooperative - Max: {coop_max:.3f}, Avg: {coop_avg:.3f}")
    
    if comp_max > coop_max:
        print("   🏆 Competitive training achieved higher peak performance")
    else:
        print("   🤝 Cooperative training achieved higher peak performance")
    
    if comp_avg > coop_avg:
        print("   📈 Competitive training achieved higher average performance")
    else:
        print("   📈 Cooperative training achieved higher average performance")


if __name__ == "__main__":
    print("=" * 60)
    print("⚔️ Core SRL - Competitive Multi-Model Training")
    print("=" * 60)
    
    # Run competitive training
    results = asyncio.run(competitive_training_example())
    
    # Run comparison
    asyncio.run(competitive_vs_cooperative_comparison())
    
    print("\n" + "=" * 60)
    print("✨ Competitive training example completed!")
    print("💡 Key insights:")
    print("   - Different model types show different competitive advantages")
    print("   - Selective weight updates create performance gaps")
    print("   - Competition drives specialization and diversity")
    print("=" * 60)
