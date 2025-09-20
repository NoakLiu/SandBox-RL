#!/usr/bin/env python3
"""
Checkpoint Management Example
============================

Comprehensive example showing how to save, load, and manage training checkpoints
for multi-model RL training sessions.
"""

import asyncio
import logging
import os
import json
from pathlib import Path
from core_srl import (
    MultiModelTrainer,
    MultiModelConfig,
    TrainingMode,
    list_available_checkpoints,
    load_checkpoint_metadata
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def checkpoint_training_example():
    """Example showing training with checkpoint management"""
    
    print(" Checkpoint Management Training Example")
    print("=" * 50)
    
    # Configure training with checkpointing
    config = MultiModelConfig(
        num_models=4,
        model_types=["qwen3"] * 4,
        training_mode=TrainingMode.MIXED,
        max_episodes=150,
        checkpoint_dir="./checkpoints/example_training",
        save_interval=25,  # Save every 25 episodes
        max_checkpoints=5  # Keep only 5 latest checkpoints
    )
    
    trainer = MultiModelTrainer(config)
    
    print(f" Checkpoint directory: {config.checkpoint_dir}")
    print(f" Save interval: {config.save_interval} episodes")
    print(f"🗂️ Max checkpoints: {config.max_checkpoints}")
    
    try:
        # Start training with automatic checkpointing
        print("\n Starting training with automatic checkpointing...")
        
        # Simulate interruption after 50 episodes
        trainer.config.max_episodes = 50
        partial_results = await trainer.train()
        
        print(f"\n⏸️ Training interrupted at episode {trainer.current_episode}")
        
        # Check saved checkpoints
        checkpoints = list_available_checkpoints(config.checkpoint_dir)
        print(f" Checkpoints saved: {len(checkpoints)}")
        
        if checkpoints:
            latest_checkpoint = checkpoints[0]
            print(f" Latest checkpoint: {latest_checkpoint}")
            
            # Load checkpoint metadata
            metadata = load_checkpoint_metadata(latest_checkpoint, config.checkpoint_dir)
            if metadata:
                print(f" Checkpoint episode: {metadata['episode']}")
                print(f"📅 Saved at: {metadata['timestamp']}")
                print(f" Models: {len(metadata['model_states'])}")
        
        await trainer.shutdown()
        
        return checkpoints
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        await trainer.shutdown()
        return []


async def resume_training_example():
    """Example showing how to resume training from checkpoint"""
    
    print("\n Resume Training Example")
    print("=" * 30)
    
    checkpoint_dir = "./checkpoints/example_training"
    
    # Check available checkpoints
    checkpoints = list_available_checkpoints(checkpoint_dir)
    
    if not checkpoints:
        print("❌ No checkpoints found. Run checkpoint_training_example() first.")
        return
    
    print(f" Found {len(checkpoints)} checkpoints")
    
    # Load latest checkpoint
    latest_checkpoint = checkpoints[0]
    metadata = load_checkpoint_metadata(latest_checkpoint, checkpoint_dir)
    
    print(f" Loading checkpoint: {latest_checkpoint}")
    print(f" Previous episode: {metadata['episode']}")
    
    # Create new trainer and load checkpoint
    config = MultiModelConfig(
        num_models=4,
        model_types=["qwen3"] * 4,
        training_mode=TrainingMode.MIXED,
        max_episodes=100,  # Continue for 100 more episodes
        checkpoint_dir=checkpoint_dir
    )
    
    trainer = MultiModelTrainer(config)
    
    try:
        # Load checkpoint
        success = trainer.load_checkpoint(latest_checkpoint)
        
        if success:
            print(f"✅ Successfully resumed from episode {trainer.current_episode}")
            
            # Continue training
            print(" Continuing training...")
            results = await trainer.train()
            
            print(f"\n✅ Training completed!")
            print(f" Final episode: {results['total_episodes']}")
            
            # Show progress
            final_performance = trainer.get_model_performance_summary()
            for model_id, stats in final_performance.items():
                print(f"   {model_id}: {stats['avg_reward']:.3f} final reward")
            
            return results
        else:
            print("❌ Failed to load checkpoint")
            return None
            
    finally:
        await trainer.shutdown()


async def checkpoint_analysis_example():
    """Analyze training progress across multiple checkpoints"""
    
    print("\n Checkpoint Analysis Example")
    print("=" * 35)
    
    checkpoint_dir = "./checkpoints/example_training"
    checkpoints = list_available_checkpoints(checkpoint_dir)
    
    if len(checkpoints) < 2:
        print("❌ Need at least 2 checkpoints for analysis")
        return
    
    print(f" Analyzing {len(checkpoints)} checkpoints...")
    
    # Load all checkpoint metadata
    checkpoint_data = []
    for checkpoint_id in reversed(checkpoints):  # Oldest to newest
        metadata = load_checkpoint_metadata(checkpoint_id, checkpoint_dir)
        if metadata:
            checkpoint_data.append(metadata)
    
    # Analyze training progress
    episodes = [data['episode'] for data in checkpoint_data]
    
    # Calculate progress metrics for each model
    model_progress = {}
    
    for data in checkpoint_data:
        episode = data['episode']
        for model_id, state in data['model_states'].items():
            if model_id not in model_progress:
                model_progress[model_id] = {
                    'episodes': [],
                    'rewards': [],
                    'win_rates': [],
                    'update_counts': []
                }
            
            avg_reward = state['total_reward'] / max(1, state['episode_count'])
            win_rate = state['win_count'] / max(1, state['episode_count'])
            
            model_progress[model_id]['episodes'].append(episode)
            model_progress[model_id]['rewards'].append(avg_reward)
            model_progress[model_id]['win_rates'].append(win_rate)
            model_progress[model_id]['update_counts'].append(state['update_count'])
    
    # Show progress analysis
    print(f"\n Training Progress Analysis:")
    
    for model_id, progress in model_progress.items():
        if len(progress['rewards']) >= 2:
            initial_reward = progress['rewards'][0]
            final_reward = progress['rewards'][-1]
            improvement = final_reward - initial_reward
            
            print(f"   {model_id}:")
            print(f"     Initial Reward: {initial_reward:.3f}")
            print(f"     Final Reward: {final_reward:.3f}")
            print(f"     Improvement: {improvement:+.3f}")
            print(f"     Updates: {progress['update_counts'][-1]}")
    
    # Calculate overall training effectiveness
    all_improvements = []
    for progress in model_progress.values():
        if len(progress['rewards']) >= 2:
            improvement = progress['rewards'][-1] - progress['rewards'][0]
            all_improvements.append(improvement)
    
    if all_improvements:
        avg_improvement = sum(all_improvements) / len(all_improvements)
        print(f"\n Overall Training Effectiveness:")
        print(f"   Average Improvement: {avg_improvement:+.3f}")
        
        if avg_improvement > 0.1:
            print("   ✅ Excellent training progress")
        elif avg_improvement > 0.05:
            print("   ✅ Good training progress")
        else:
            print("   ⚠️ Limited training progress")


async def checkpoint_cleanup_example():
    """Example showing checkpoint cleanup and management"""
    
    print("\n Checkpoint Cleanup Example")
    print("=" * 32)
    
    checkpoint_dir = "./checkpoints/example_training"
    checkpoints = list_available_checkpoints(checkpoint_dir)
    
    print(f" Found {len(checkpoints)} checkpoints")
    
    # Calculate total size
    total_size = 0
    for checkpoint_id in checkpoints:
        checkpoint_path = Path(checkpoint_dir) / checkpoint_id
        if checkpoint_path.exists():
            for file_path in checkpoint_path.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
    
    total_size_mb = total_size / (1024 * 1024)
    print(f"💽 Total size: {total_size_mb:.2f} MB")
    
    # Show checkpoint details
    print(f"\n Checkpoint Details:")
    for i, checkpoint_id in enumerate(checkpoints):
        metadata = load_checkpoint_metadata(checkpoint_id, checkpoint_dir)
        if metadata:
            episode = metadata['episode']
            timestamp = metadata['timestamp']
            is_final = checkpoint_id.startswith('final_')
            
            status = " Final" if is_final else f" Episode {episode}"
            print(f"   {i+1}. {checkpoint_id}")
            print(f"      {status} - {timestamp}")
    
    # Cleanup old checkpoints (keep only final and latest 3)
    print(f"\n Cleanup Strategy:")
    
    final_checkpoints = [c for c in checkpoints if c.startswith('final_')]
    regular_checkpoints = [c for c in checkpoints if not c.startswith('final_')]
    
    # Keep all final checkpoints + 3 latest regular
    keep_checkpoints = final_checkpoints + regular_checkpoints[:3]
    cleanup_checkpoints = regular_checkpoints[3:]
    
    print(f"   Keep: {len(keep_checkpoints)} checkpoints")
    print(f"   Cleanup: {len(cleanup_checkpoints)} checkpoints")
    
    if cleanup_checkpoints:
        print(f"   Would cleanup: {cleanup_checkpoints}")
        # Note: Actual cleanup would require shutil.rmtree()


if __name__ == "__main__":
    print("=" * 60)
    print(" Core SRL - Checkpoint Management Examples")
    print("=" * 60)
    
    # Run checkpoint training
    checkpoints = asyncio.run(checkpoint_training_example())
    
    if checkpoints:
        # Resume training example
        asyncio.run(resume_training_example())
        
        # Analysis example
        asyncio.run(checkpoint_analysis_example())
        
        # Cleanup example
        asyncio.run(checkpoint_cleanup_example())
    
    print("\n" + "=" * 60)
    print(" Checkpoint management examples completed!")
    print(" Key takeaways:")
    print("   - Checkpoints are saved automatically during training")
    print("   - Training can be resumed seamlessly from any checkpoint")
    print("   - Checkpoint analysis helps understand training progress")
    print("   - Cleanup prevents disk space issues")
    print("=" * 60)
