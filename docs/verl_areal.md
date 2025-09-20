# VERL/AReaL Integration Guide

Advanced optimization with VERL (Versatile Efficient RL) and AReaL frameworks for high-performance multi-model training.

## Overview

Core SRL integrates two cutting-edge frameworks:
- **VERL**: Efficient RL training with vLLM backend and distributed rollouts
- **AReaL**: Advanced caching, resource optimization, and memory management

## VERL Integration

### Enable VERL Training

```python
from core_srl import MultiModelConfig, MultiModelTrainer

config = MultiModelConfig(
    enable_verl=True,
    num_models=4,
    model_names={
        "qwen3": "Qwen/Qwen2.5-14B-Instruct"
    }
)

trainer = MultiModelTrainer(config)
```

### VERL Configuration

```python
from core_srl import create_verl_trainer

# Create VERL trainer for specific model
verl_trainer = create_verl_trainer("Qwen/Qwen2.5-14B-Instruct")

# Configure for multi-model scenario
verl_config = {
    "batch_size": 32,
    "learning_rate": 3e-4,
    "num_workers": 4,
    "vllm_url": "http://localhost:8001"
}
```

### VERL Training Loop

```python
async def verl_training_example():
    from core_srl import VERLTrainer
    
    trainer = VERLTrainer("Qwen/Qwen2.5-14B-Instruct")
    
    prompts = [
        "Solve this problem cooperatively:",
        "Analyze the competitive scenario:",
        "What is the optimal strategy?"
    ]
    
    # VERL rollout step
    rollout_data = await trainer.rollout_step(prompts)
    print(f"Rewards: {rollout_data['rewards']}")
    
    # VERL training step
    train_result = await trainer.train_step(rollout_data)
    print(f"Losses: {train_result['losses']}")
    
    return train_result
```

## AReaL Integration

### Enable AReaL Optimization

```python
config = MultiModelConfig(
    enable_areal=True,
    kv_cache_size=20000,  # Larger cache for AReaL
    enable_verl=True      # Best used together
)
```

### AReaL Cache Management

```python
from core_srl import create_areal_integration

# Create AReaL integration manager
areal_manager = create_areal_integration(
    cache_size=15000,
    max_memory_gb=16.0
)

# Get cache statistics
cache_stats = areal_manager.get_stats()
print(f"Cache hit rate: {cache_stats['cache_stats']['hit_rate']:.3f}")
```

### KV Cache Optimization

```python
from core_srl import create_kv_cache_manager, CachePolicy

# Create optimized KV cache
kv_cache = create_kv_cache_manager(
    max_cache_size=20000,
    cache_policy=CachePolicy.ADAPTIVE
)

# Monitor cache performance
stats = kv_cache.get_stats()
print(f"Cache efficiency: {stats['hit_rate']:.3f}")
print(f"Memory usage: {stats['memory_usage']:.2f} GB")
```

## Combined VERL/AReaL Training

### Integrated Training Bridge

```python
from core_srl import create_areal_verl_bridge

# Create integrated trainer
bridge = create_areal_verl_bridge("Qwen/Qwen2.5-14B-Instruct")

# Run integrated training
prompts = ["Training prompt 1", "Training prompt 2"]
results = await bridge.integrated_training_loop(
    prompts=prompts,
    num_steps=100
)

print(f"Training metrics: {results['training_metrics']}")
print(f"Final stats: {results['final_stats']}")
```

### Multi-Model VERL/AReaL

```python
class VERLAReaLMultiModelTrainer(MultiModelTrainer):
    def __init__(self, config):
        super().__init__(config)
        
        # Initialize VERL/AReaL for each model
        self.verl_trainers = {}
        self.areal_bridges = {}
        
        for model_id in self.llm_managers.keys():
            model_name = self.model_states[model_id].model_name
            
            # Create VERL trainer for this model
            self.verl_trainers[model_id] = VERLTrainer(model_name)
            
            # Create AReaL bridge
            self.areal_bridges[model_id] = create_areal_verl_bridge(model_name)
    
    async def train_with_optimization(self):
        """Training with VERL/AReaL optimization"""
        for episode in range(self.config.max_episodes):
            # Standard multi-model episode
            episode_result = await self.train_multi_model_episode(episode)
            
            # Apply VERL/AReaL optimization
            optimization_results = {}
            
            for model_id, bridge in self.areal_bridges.items():
                prompts = [f"Optimize episode {episode} for {model_id}"]
                opt_result = await bridge.integrated_training_loop(prompts, 1)
                optimization_results[model_id] = opt_result
            
            # Log optimization impact
            if episode % 50 == 0:
                self._log_optimization_metrics(optimization_results)
```

## Performance Optimization

### Cache Tuning

```python
# Tune cache for your hardware
def tune_cache_config(available_memory_gb):
    if available_memory_gb >= 32:
        return {
            "kv_cache_size": 25000,
            "max_memory_gb": 24.0,
            "cache_policy": CachePolicy.ADAPTIVE
        }
    elif available_memory_gb >= 16:
        return {
            "kv_cache_size": 15000,
            "max_memory_gb": 12.0,
            "cache_policy": CachePolicy.LRU
        }
    else:
        return {
            "kv_cache_size": 8000,
            "max_memory_gb": 6.0,
            "cache_policy": CachePolicy.FIFO
        }

# Apply tuning
cache_config = tune_cache_config(32)  # 32GB available
config.kv_cache_size = cache_config["kv_cache_size"]
```

### Memory Management

```python
# Monitor memory usage during training
class MemoryOptimizedTrainer(MultiModelTrainer):
    async def train_multi_model_episode(self, episode_num):
        # Check memory before episode
        if self._get_memory_usage() > 0.9:  # >90% memory usage
            await self._optimize_memory()
        
        # Run episode
        result = await super().train_multi_model_episode(episode_num)
        
        # Cleanup after episode
        if episode_num % 10 == 0:
            await self._cleanup_cache()
        
        return result
    
    def _get_memory_usage(self):
        """Get current memory usage ratio"""
        try:
            import psutil
            return psutil.virtual_memory().percent / 100.0
        except ImportError:
            return 0.5  # Assume 50% if can't measure
    
    async def _optimize_memory(self):
        """Optimize memory usage"""
        # Clear old cache entries
        if hasattr(self, 'verl_areal_bridge'):
            cache = self.verl_areal_bridge.areal_manager.get_cache()
            if cache:
                cache.clear()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        print("Memory optimization completed")
```

## Integration Examples

### Full Integration Example

```python
import asyncio
from core_srl import (
    MultiModelTrainer, 
    MultiModelConfig,
    TrainingMode,
    WeightUpdateStrategy
)

async def full_verl_areal_training():
    # Configure with all optimizations
    config = MultiModelConfig(
        num_models=4,
        model_types=["qwen3", "openai"],
        training_mode=TrainingMode.MIXED,
        weight_update_strategy=WeightUpdateStrategy.FEDERATED,
        
        # VERL/AReaL optimization
        enable_verl=True,
        enable_areal=True,
        kv_cache_size=20000,
        
        # Training parameters
        max_episodes=500,
        learning_rate=3e-4,
        batch_size=32,
        
        # Checkpointing
        save_interval=50,
        checkpoint_dir="./optimized_training"
    )
    
    # Create optimized trainer
    trainer = MultiModelTrainer(config)
    
    try:
        print("Starting VERL/AReaL optimized training...")
        results = await trainer.train()
        
        # Analyze optimization impact
        final_stats = results['training_metrics']
        avg_throughput = sum(final_stats.get('throughput', [1.0])) / len(final_stats.get('throughput', [1.0]))
        
        print(f"Training completed with {avg_throughput:.2f} episodes/sec average throughput")
        
        return results
        
    finally:
        await trainer.shutdown()

# Run optimized training
results = asyncio.run(full_verl_areal_training())
```

### Performance Monitoring

```python
# Monitor VERL/AReaL performance
def monitor_optimization_performance(trainer):
    """Monitor optimization metrics during training"""
    
    if hasattr(trainer, 'verl_areal_bridge'):
        bridge = trainer.verl_areal_bridge
        
        # Get VERL stats
        verl_stats = bridge.get_training_summary()
        
        # Get AReaL stats  
        areal_stats = bridge.areal_manager.get_stats()
        
        print("Optimization Performance:")
        print(f"  Cache hit rate: {areal_stats.get('cache_stats', {}).get('hit_rate', 0):.3f}")
        print(f"  VERL steps: {verl_stats.get('verl_stats', {}).get('step_count', 0)}")
        
        return {
            "verl_stats": verl_stats,
            "areal_stats": areal_stats
        }
```

## Troubleshooting

### Common Issues

**High Memory Usage:**
```python
# Reduce cache size
config.kv_cache_size = 5000
# Or disable some optimizations
config.enable_areal = False
```

**Slow Training:**
```python
# Increase cache size
config.kv_cache_size = 30000
# Enable all optimizations
config.enable_verl = True
config.enable_areal = True
```

**Cache Misses:**
```python
# Use adaptive cache policy
from core_srl import CachePolicy
cache_manager = create_kv_cache_manager(cache_policy=CachePolicy.ADAPTIVE)
```

The VERL/AReaL integration provides significant performance improvements for multi-model training scenarios, especially with larger models and longer training runs.
