#!/usr/bin/env python3
"""
Distributed Multi-Model Training Example
========================================

Example showing distributed training across multiple GPUs with
resource management, load balancing, and coordination.
"""

import asyncio
import logging
from core_srl import (
    MultiModelTrainer,
    MultiModelConfig,
    TrainingMode,
    WeightUpdateStrategy,
    create_distributed_lora_scheduler,
    create_unified_scheduler
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def distributed_gpu_training():
    """Distributed training across multiple GPUs"""
    
    print(" Distributed GPU Training Example")
    print("=" * 38)
    
    # Configure for 8 models across 4 GPUs
    config = MultiModelConfig(
        num_models=8,
        num_gpus=4,  # 2 models per GPU
        model_types=["qwen3"] * 8,
        model_names={"qwen3": "Qwen/Qwen2.5-14B-Instruct"},
        training_mode=TrainingMode.MIXED,
        weight_update_strategy=WeightUpdateStrategy.FEDERATED,
        base_port=8001,  # Ports 8001-8008
        max_episodes=200,
        batch_size=64,  # Larger batch for distributed training
        checkpoint_dir="./checkpoints/distributed"
    )
    
    trainer = MultiModelTrainer(config)
    
    print(f" Distributed Configuration:")
    print(f"   Models: {config.num_models}")
    print(f"   GPUs: {config.num_gpus}")
    print(f"   Models per GPU: {config.num_models // config.num_gpus}")
    print(f"   Port range: {config.base_port}-{config.base_port + config.num_models - 1}")
    
    try:
        # Check GPU allocation
        print(f"\n GPU Allocation:")
        for model_id, state in trainer.model_states.items():
            print(f"   {model_id} → GPU {state.gpu_id}")
        
        # Start distributed training
        print(f"\n Starting distributed training...")
        
        # Monitor resource usage during training
        resource_history = []
        
        for episode in range(0, 200, 10):  # Sample every 10 episodes
            # Train 10 episodes
            for ep in range(episode, min(episode + 10, 200)):
                episode_result = await trainer.train_multi_model_episode(ep)
            
            # Collect resource stats
            scheduler_stats = trainer.scheduler.get_system_statistics()
            resource_stats = scheduler_stats.get('resource_statistics', {})
            
            resource_history.append({
                'episode': episode + 10,
                'gpu_allocation_rates': resource_stats.get('gpu_allocation_rates', {}),
                'total_competitions': resource_stats.get('total_competitions', 0)
            })
            
            print(f"   Episode {episode + 10}: Resource utilization logged")
        
        print(f"\n✅ Distributed training completed!")
        
        # Analyze resource utilization
        final_stats = trainer.scheduler.get_system_statistics()
        
        print(f"\n Resource Utilization Analysis:")
        resource_stats = final_stats.get('resource_statistics', {})
        
        if 'gpu_allocation_rates' in resource_stats:
            for gpu_id, rates in resource_stats['gpu_allocation_rates'].items():
                compute_rate = rates.get('compute_rate', 0.0)
                memory_rate = rates.get('memory_rate', 0.0)
                print(f"   {gpu_id}: Compute={compute_rate:.3f}, Memory={memory_rate:.3f}")
        
        # Check for resource competition
        total_competitions = resource_stats.get('total_competitions', 0)
        print(f"   Resource competitions: {total_competitions}")
        
        if total_competitions > 50:
            print("   ⚠️ High resource contention - consider more GPUs")
        elif total_competitions > 10:
            print("   ✅ Moderate resource usage - good balance")
        else:
            print("   ✅ Low resource contention - efficient utilization")
        
        # Model performance across GPUs
        performance = trainer.get_model_performance_summary()
        gpu_performance = {}
        
        for model_id, stats in performance.items():
            gpu_id = stats['gpu_id']
            if gpu_id not in gpu_performance:
                gpu_performance[gpu_id] = []
            gpu_performance[gpu_id].append(stats['avg_reward'])
        
        print(f"\n Performance by GPU:")
        for gpu_id, rewards in gpu_performance.items():
            avg_reward = sum(rewards) / len(rewards)
            print(f"   GPU {gpu_id}: {avg_reward:.3f} average reward ({len(rewards)} models)")
        
        return final_stats
        
    finally:
        await trainer.shutdown()


async def lora_distributed_training():
    """Distributed training with LoRA management"""
    
    print("\n Distributed LoRA Training Example")
    print("=" * 38)
    
    # Setup distributed LoRA scheduler
    lora_scheduler = create_distributed_lora_scheduler(
        base_port=8001,
        num_gpus=4,
        model_name="Qwen/Qwen2.5-14B-Instruct"
    )
    
    await lora_scheduler.start()
    
    print(" Distributed LoRA scheduler started")
    print(f"   GPUs: 4")
    print(f"   LoRA adapters: 8 (2 per GPU)")
    
    # Configure training with LoRA
    config = MultiModelConfig(
        num_models=8,
        num_gpus=4,
        model_types=["qwen3"] * 8,
        training_mode=TrainingMode.COMPETITIVE,
        max_episodes=100,
        enable_monitoring=True
    )
    
    trainer = MultiModelTrainer(config)
    
    try:
        print(f"\n Starting LoRA-enabled distributed training...")
        
        # Monitor LoRA performance
        lora_stats_history = []
        
        for episode in range(0, 100, 20):
            # Train batch of episodes
            for ep in range(episode, min(episode + 20, 100)):
                episode_result = await trainer.train_multi_model_episode(ep)
                
                # Simulate LoRA updates based on rewards
                for model_id, reward in episode_result['rewards'].items():
                    if reward > 1.0:  # Good performance
                        lora_id = int(model_id.split('_')[1]) + 1
                        try:
                            # Simulate LoRA weight update
                            new_weights = {"rank": 16, "alpha": 32.0}
                            await lora_scheduler.submit_rl_update(lora_id, reward, new_weights)
                        except Exception as e:
                            logger.warning(f"LoRA update failed for {lora_id}: {e}")
            
            # Collect LoRA stats
            system_status = await lora_scheduler.get_system_status()
            lora_stats_history.append({
                'episode': episode + 20,
                'lora_status': system_status.get('lora_status', {}),
                'training_stats': system_status.get('training_stats', {})
            })
            
            print(f"   Episode {episode + 20}: LoRA stats collected")
        
        print(f"\n✅ LoRA distributed training completed!")
        
        # Analyze LoRA performance
        final_lora_stats = lora_stats_history[-1] if lora_stats_history else {}
        lora_status = final_lora_stats.get('lora_status', {})
        
        print(f"\n LoRA Performance Summary:")
        
        successful_loras = 0
        total_updates = 0
        
        for lora_id, status in lora_status.items():
            if status.get('loading_status') == 'ready':
                successful_loras += 1
            total_updates += status.get('update_count', 0)
        
        print(f"   Ready LoRAs: {successful_loras}/{len(lora_status)}")
        print(f"   Total updates: {total_updates}")
        print(f"   Average updates per LoRA: {total_updates/max(len(lora_status), 1):.1f}")
        
        # Training effectiveness with LoRA
        training_stats = final_lora_stats.get('training_stats', {})
        if training_stats:
            print(f"   Training updates: {training_stats.get('total_updates', 0)}")
            
            reward_stats = training_stats.get('reward_stats', {})
            if reward_stats:
                print(f"   Reward range: {reward_stats.get('min', 0):.3f} - {reward_stats.get('max', 0):.3f}")
                print(f"   Average reward: {reward_stats.get('avg', 0):.3f}")
        
        return system_status
        
    finally:
        await lora_scheduler.stop()
        await trainer.shutdown()


async def load_balancing_analysis():
    """Analyze load balancing across distributed setup"""
    
    print("\n Load Balancing Analysis")
    print("=" * 28)
    
    # Setup distributed training
    config = MultiModelConfig(
        num_models=8,
        num_gpus=4,
        training_mode=TrainingMode.MIXED,
        max_episodes=100
    )
    
    trainer = MultiModelTrainer(config)
    
    try:
        print(" Analyzing load distribution...")
        
        # Run training and collect load stats
        load_stats = []
        
        for episode in range(0, 100, 25):
            # Train batch
            for ep in range(episode, min(episode + 25, 100)):
                await trainer.train_multi_model_episode(ep)
            
            # Collect load statistics
            scheduler_stats = trainer.scheduler.get_system_statistics()
            model_stats = scheduler_stats.get('model_statistics', {})
            
            gpu_distribution = model_stats.get('gpu_distribution', {})
            load_stats.append({
                'episode': episode + 25,
                'gpu_distribution': gpu_distribution
            })
            
            print(f"   Episode {episode + 25}: Load stats collected")
        
        # Analyze load balance
        final_distribution = load_stats[-1]['gpu_distribution'] if load_stats else {}
        
        print(f"\n Final Load Distribution:")
        
        gpu_loads = []
        for gpu_id, model_count in final_distribution.items():
            print(f"   {gpu_id}: {model_count} models")
            gpu_loads.append(model_count)
        
        if gpu_loads:
            max_load = max(gpu_loads)
            min_load = min(gpu_loads)
            load_imbalance = (max_load - min_load) / max(max_load, 1)
            
            print(f"\n Load Balance Analysis:")
            print(f"   Load imbalance: {load_imbalance:.3f}")
            
            if load_imbalance < 0.2:
                print("   ✅ Excellent load balance")
            elif load_imbalance < 0.4:
                print("   ✅ Good load balance")
            else:
                print("   ⚠️ Poor load balance - consider rebalancing")
        
        return load_stats
        
    finally:
        await trainer.shutdown()


if __name__ == "__main__":
    print("=" * 60)
    print(" Core SRL - Distributed Training Examples")
    print("=" * 60)
    
    # Run distributed GPU training
    distributed_stats = asyncio.run(distributed_gpu_training())
    
    # Run LoRA distributed training
    lora_stats = asyncio.run(lora_distributed_training())
    
    # Analyze load balancing
    load_stats = asyncio.run(load_balancing_analysis())
    
    print("\n" + "=" * 60)
    print(" Distributed training examples completed!")
    print(" Key insights:")
    print("   - Distributed training scales well across GPUs")
    print("   - LoRA enables efficient parameter updates")
    print("   - Load balancing is crucial for performance")
    print("   - Resource competition indicates high utilization")
    print("=" * 60)
