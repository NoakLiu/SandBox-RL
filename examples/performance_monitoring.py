#!/usr/bin/env python3
"""
Performance Monitoring Example
==============================

Example showing real-time performance monitoring during multi-model training
with metrics collection, visualization, and analysis.
"""

import asyncio
import logging
import time
from core_srl import (
    MultiModelTrainer,
    MultiModelConfig,
    TrainingMode,
    create_unified_monitor,
    create_social_network_metrics,
    MonitoringConfig
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def basic_monitoring_example():
    """Basic monitoring during training"""
    
    print(" Basic Performance Monitoring Example")
    print("=" * 42)
    
    # Setup monitoring
    monitor_config = MonitoringConfig(
        enable_file_logging=True,
        log_file_path="./logs/training_metrics.json",
        metrics_sampling_interval=1.0
    )
    
    monitor = create_unified_monitor(monitor_config)
    monitor.start()
    
    print(" Monitoring system started")
    print(f" Log file: {monitor_config.log_file_path}")
    
    # Configure training with monitoring
    config = MultiModelConfig(
        num_models=4,
        training_mode=TrainingMode.MIXED,
        max_episodes=100,
        enable_monitoring=True,
        log_metrics_interval=5  # Log every 5 episodes
    )
    
    trainer = MultiModelTrainer(config)
    
    try:
        print("\n Starting monitored training...")
        
        # Custom monitoring during training
        monitoring_data = []
        
        for episode in range(100):
            # Train episode
            episode_result = await trainer.train_multi_model_episode(episode)
            
            # Create custom metrics
            metrics = create_social_network_metrics(
                total_users=config.num_models,
                active_users=len([m for m in trainer.model_states.values() if m.episode_count > 0]),
                engagement_rate=episode_result['cooperation_ratio'],
                response_time_avg=episode_result['episode_time'],
                avg_influence_score=episode_result['total_reward'] / config.num_models,
                network_density=episode_result['cooperation_ratio'],
                viral_spread_rate=episode_result['total_reward'] / 10.0
            )
            
            # Update monitor
            monitor.update_metrics(metrics)
            
            # Collect custom monitoring data
            monitoring_data.append({
                'episode': episode,
                'total_reward': episode_result['total_reward'],
                'cooperation_ratio': episode_result['cooperation_ratio'],
                'episode_time': episode_result['episode_time'],
                'model_rewards': episode_result['rewards']
            })
            
            # Log progress every 20 episodes
            if episode % 20 == 0:
                print(f"   Episode {episode}: reward={episode_result['total_reward']:.2f}, "
                      f"coop={episode_result['cooperation_ratio']:.3f}, "
                      f"time={episode_result['episode_time']:.3f}s")
        
        print("\n✅ Monitored training completed!")
        
        # Get comprehensive monitoring stats
        final_stats = monitor.get_comprehensive_stats()
        
        print(f"\n Final Monitoring Stats:")
        print(f"   Metrics collected: {final_stats['metrics_summary']['total_samples']}")
        print(f"   Average engagement: {final_stats['metrics_summary']['recent_avg_engagement']:.3f}")
        print(f"   Average response time: {final_stats['metrics_summary']['recent_avg_response_time']:.3f}s")
        
        # Analyze training trends
        if len(monitoring_data) >= 10:
            early_rewards = [d['total_reward'] for d in monitoring_data[:10]]
            late_rewards = [d['total_reward'] for d in monitoring_data[-10:]]
            
            early_avg = sum(early_rewards) / len(early_rewards)
            late_avg = sum(late_rewards) / len(late_rewards)
            improvement = late_avg - early_avg
            
            print(f"\n Training Progress:")
            print(f"   Early average reward: {early_avg:.3f}")
            print(f"   Late average reward: {late_avg:.3f}")
            print(f"   Improvement: {improvement:+.3f}")
            
            if improvement > 0.1:
                print("   ✅ Strong learning progress")
            elif improvement > 0.05:
                print("   ✅ Good learning progress")
            else:
                print("   ⚠️ Limited learning progress")
        
        return monitoring_data
        
    finally:
        monitor.stop()
        await trainer.shutdown()


async def real_time_visualization_example():
    """Example with real-time graph visualization"""
    
    print("\n Real-time Visualization Example")
    print("=" * 37)
    
    # Create monitor with graph visualization
    monitor = create_unified_monitor()
    monitor.start()
    
    # Create visualization scenario
    monitor.create_visualization_scenario(num_agents=8)
    
    print(" Created visualization scenario with 8 agents")
    print(" Graph visualization includes:")
    print("   - Cooperation/competition edges")
    print("   - Information spread dynamics")
    print("   - Real-time belief updates")
    
    # Configure training
    config = MultiModelConfig(
        num_models=4,
        training_mode=TrainingMode.MIXED,
        max_episodes=50,
        enable_monitoring=True
    )
    
    trainer = MultiModelTrainer(config)
    
    try:
        print("\n Starting training with visualization...")
        
        # Training with visualization updates
        for episode in range(50):
            episode_result = await trainer.train_multi_model_episode(episode)
            
            # Update visualization every 5 episodes
            if episode % 5 == 0:
                # Simulate graph interactions based on training results
                cooperation_ratio = episode_result['cooperation_ratio']
                
                if cooperation_ratio > 0.6:
                    # High cooperation - simulate cooperative interactions
                    from core_srl import InteractionType as GraphInteractionType
                    monitor.simulate_graph_interaction("user_1", "user_2", GraphInteractionType.COOPERATE)
                    monitor.simulate_graph_interaction("user_2", "user_3", GraphInteractionType.COOPERATE)
                else:
                    # Low cooperation - simulate competitive interactions
                    monitor.simulate_graph_interaction("user_1", "user_3", GraphInteractionType.COMPETE)
                
                print(f"   Episode {episode}: Updated graph visualization (coop={cooperation_ratio:.3f})")
        
        # Get final visualization stats
        viz_stats = monitor.graph_visualizer.get_statistics()
        
        print(f"\n Visualization Summary:")
        print(f"   Total nodes: {viz_stats['total_nodes']}")
        print(f"   Total edges: {viz_stats['total_edges']}")
        print(f"   Cooperation edges: {viz_stats.get('cooperation_count', 0)}")
        print(f"   Competition edges: {viz_stats.get('competition_count', 0)}")
        print(f"   Average belief: {viz_stats['average_belief']:.3f}")
        
        return viz_stats
        
    finally:
        monitor.stop()
        await trainer.shutdown()


async def performance_comparison_example():
    """Compare performance with and without monitoring"""
    
    print("\n Performance Impact Analysis")
    print("=" * 34)
    
    episodes = 50
    
    # Test without monitoring
    print("🔇 Testing without monitoring...")
    config_no_monitor = MultiModelConfig(
        num_models=3,
        max_episodes=episodes,
        enable_monitoring=False
    )
    
    trainer_no_monitor = MultiModelTrainer(config_no_monitor)
    
    start_time = time.time()
    results_no_monitor = await trainer_no_monitor.train()
    time_no_monitor = time.time() - start_time
    await trainer_no_monitor.shutdown()
    
    # Test with monitoring
    print("\n Testing with monitoring...")
    config_with_monitor = MultiModelConfig(
        num_models=3,
        max_episodes=episodes,
        enable_monitoring=True,
        log_metrics_interval=5
    )
    
    trainer_with_monitor = MultiModelTrainer(config_with_monitor)
    
    start_time = time.time()
    results_with_monitor = await trainer_with_monitor.train()
    time_with_monitor = time.time() - start_time
    await trainer_with_monitor.shutdown()
    
    # Compare results
    print(f"\n Performance Comparison:")
    print(f"   Without monitoring: {time_no_monitor:.2f}s ({episodes/time_no_monitor:.2f} eps/sec)")
    print(f"   With monitoring: {time_with_monitor:.2f}s ({episodes/time_with_monitor:.2f} eps/sec)")
    
    overhead = (time_with_monitor - time_no_monitor) / time_no_monitor * 100
    print(f"   Monitoring overhead: {overhead:+.1f}%")
    
    if overhead < 10:
        print("   ✅ Low overhead - monitoring recommended")
    elif overhead < 25:
        print("   ⚠️ Moderate overhead - consider reducing monitoring frequency")
    else:
        print("   ❌ High overhead - consider disabling some monitoring features")


if __name__ == "__main__":
    print("=" * 60)
    print(" Core SRL - Performance Monitoring Examples")
    print("=" * 60)
    
    # Run basic monitoring
    monitoring_data = asyncio.run(basic_monitoring_example())
    
    # Run visualization example
    viz_stats = asyncio.run(real_time_visualization_example())
    
    # Run performance comparison
    asyncio.run(performance_comparison_example())
    
    print("\n" + "=" * 60)
    print(" Performance monitoring examples completed!")
    print(" Key insights:")
    print("   - Real-time monitoring provides valuable training insights")
    print("   - Monitoring overhead is typically <10%")
    print("   - Graph visualization helps understand model interactions")
    print("   - Metrics help optimize training parameters")
    print("=" * 60)
