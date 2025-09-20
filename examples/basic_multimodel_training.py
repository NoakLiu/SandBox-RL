#!/usr/bin/env python3
"""
Basic Multi-Model Training Example
=================================

Simple example showing how to train 4 modern LLMs together using Core SRL.
This is the recommended starting point for new users.
"""

import asyncio
import logging
from core_srl import quick_start_multimodel_training, list_available_checkpoints

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def basic_training_example():
    """Basic multi-model training with 4 modern LLMs"""
    
    print("🚀 Starting basic multi-model training...")
    print("📋 Configuration:")
    print("   - Models: 4 (Qwen3-14B)")
    print("   - Episodes: 100")
    print("   - Mode: Mixed (cooperation + competition)")
    
    try:
        # Start training with default settings
        results = await quick_start_multimodel_training(
            num_models=4,
            max_episodes=100
        )
        
        print("\n✅ Training completed successfully!")
        print(f"📊 Status: {results['status']}")
        
        # Show individual model performance
        print("\n🤖 Model Performance:")
        for model_id, performance in results['model_performance'].items():
            print(f"   {model_id}:")
            print(f"     Average Reward: {performance['avg_reward']:.3f}")
            print(f"     Win Rate: {performance['win_rate']:.3f}")
            print(f"     Episodes: {performance['total_episodes']}")
            print(f"     Updates: {performance['update_count']}")
        
        # Show training metrics
        training_results = results['training_results']
        total_time = training_results['training_time']
        total_episodes = training_results['total_episodes']
        
        print(f"\n⏱️ Training Summary:")
        print(f"   Total Time: {total_time:.2f} seconds")
        print(f"   Episodes Completed: {total_episodes}")
        print(f"   Speed: {total_episodes/total_time:.2f} episodes/second")
        
        # Check saved checkpoints
        checkpoints = list_available_checkpoints()
        print(f"\n💾 Checkpoints Saved: {len(checkpoints)}")
        if checkpoints:
            print(f"   Latest: {checkpoints[0]}")
        
        return results
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"❌ Training failed: {e}")
        return None


async def analyze_results(results):
    """Analyze training results"""
    if not results:
        return
    
    print("\n📈 Detailed Analysis:")
    
    # Calculate cooperation vs competition effectiveness
    model_performance = results['model_performance']
    rewards = [p['avg_reward'] for p in model_performance.values()]
    
    # Cooperation indicator: low variance in performance
    mean_reward = sum(rewards) / len(rewards)
    variance = sum((r - mean_reward)**2 for r in rewards) / len(rewards)
    cooperation_effectiveness = 1.0 / (1.0 + variance)  # Higher = more cooperative
    
    # Competition indicator: high max reward
    max_reward = max(rewards)
    competition_effectiveness = max_reward / mean_reward  # Higher = more competitive
    
    print(f"   Cooperation Effectiveness: {cooperation_effectiveness:.3f}")
    print(f"   Competition Effectiveness: {competition_effectiveness:.3f}")
    print(f"   Overall Performance: {mean_reward:.3f} ± {variance**0.5:.3f}")
    
    # Identify best performing model
    best_model = max(model_performance.items(), key=lambda x: x[1]['avg_reward'])
    print(f"   🏆 Best Model: {best_model[0]} ({best_model[1]['avg_reward']:.3f} reward)")


if __name__ == "__main__":
    print("=" * 60)
    print("🎯 Core SRL - Basic Multi-Model Training Example")
    print("=" * 60)
    
    # Run training
    results = asyncio.run(basic_training_example())
    
    # Analyze results
    if results:
        asyncio.run(analyze_results(results))
    
    print("\n" + "=" * 60)
    print("✨ Example completed! Check the checkpoints directory for saved models.")
    print("💡 Next: Try examples/competitive_training.py or examples/cooperative_training.py")
    print("=" * 60)
