# Multi-Model Training Guide

Complete guide for training multiple modern LLMs simultaneously with cooperative-competitive RL.

## Overview

Core SRL enables training 4-8 modern LLMs together, allowing them to:
- **Cooperate**: Share knowledge and coordinate strategies
- **Compete**: Contest for resources and performance rankings
- **Adapt**: Dynamically adjust between cooperation and competition

## Training Modes

### Cooperative Mode
Models help each other learn faster:

```python
from core_srl import create_cooperative_multimodel_trainer

trainer = create_cooperative_multimodel_trainer(num_models=4)
trainer.config.cooperation_strength = 0.8  # High cooperation
trainer.config.weight_update_strategy = WeightUpdateStrategy.SYNCHRONIZED

results = await trainer.train()
```

### Competitive Mode
Models compete for performance:

```python
from core_srl import create_competitive_multimodel_trainer

trainer = create_competitive_multimodel_trainer(num_models=4)
trainer.config.competition_intensity = 0.7  # High competition
trainer.config.weight_update_strategy = WeightUpdateStrategy.SELECTIVE

results = await trainer.train()
```

### Mixed Mode (Recommended)
Dynamic balance of cooperation and competition:

```python
from core_srl import MultiModelTrainer, MultiModelConfig, TrainingMode

config = MultiModelConfig(
    training_mode=TrainingMode.MIXED,
    cooperation_strength=0.6,
    competition_intensity=0.4,
    weight_update_strategy=WeightUpdateStrategy.FEDERATED
)

trainer = MultiModelTrainer(config)
results = await trainer.train()
```

## Weight Update Strategies

### Synchronized Updates
All models update together with averaged gradients:

```python
config.weight_update_strategy = WeightUpdateStrategy.SYNCHRONIZED
# Best for: Cooperative learning, knowledge sharing
```

### Asynchronous Updates  
Models update independently:

```python
config.weight_update_strategy = WeightUpdateStrategy.ASYNCHRONOUS
# Best for: Competitive learning, diverse strategies
```

### Federated Updates
Weighted averaging based on performance:

```python
config.weight_update_strategy = WeightUpdateStrategy.FEDERATED
# Best for: Mixed scenarios, adaptive learning
```

### Selective Updates
Only top performers update:

```python
config.weight_update_strategy = WeightUpdateStrategy.SELECTIVE
# Best for: Competitive scenarios, performance focus
```

## Model Configuration

### Modern LLM Setup

```python
config = MultiModelConfig(
    num_models=6,
    model_types=["qwen3", "openai", "claude", "llama3"],
    model_names={
        "qwen3": "Qwen/Qwen2.5-14B-Instruct",
        "openai": "gpt-4o",
        "claude": "claude-3-5-sonnet-20241022", 
        "llama3": "meta-llama/Llama-3.1-8B-Instruct"
    }
)
```

### Specialized Models

```python
# Code-focused training
config.model_names = {
    "qwen_coder": "Qwen/Qwen2.5-Coder-14B-Instruct",
    "qwen_math": "Qwen/Qwen2.5-Math-14B-Instruct",
    "openai": "gpt-4o",
    "claude": "claude-3-5-sonnet-20241022"
}
```

### GPU Configuration

```python
config = MultiModelConfig(
    num_models=8,
    num_gpus=4,  # 2 models per GPU
    base_port=8001,  # vLLM ports 8001-8008
    batch_size=32
)
```

## Training Environments

### Cooperative-Competitive Environment
Standard multi-model training:

```python
from core_srl import create_multi_model_coop_compete_env

env = create_multi_model_coop_compete_env(
    num_models=4,
    cooperation_level=0.6
)
```

### Team Battle (4v4)
Team-based competition:

```python
from core_srl import create_multi_model_team_battle

env = create_multi_model_team_battle()
# Automatically creates 2 teams of 4 models each
```

### Staged Environment
Gradual difficulty increase:

```python
from core_srl import create_multi_model_staged_env

env = create_multi_model_staged_env(
    num_models=8,
    warmup_episodes=50  # Equal rewards first 50 episodes
)
```

## Monitoring and Visualization

### Real-time Monitoring

```python
from core_srl import create_unified_monitor

monitor = create_unified_monitor()
monitor.start()

# Training automatically sends metrics to monitor
trainer.config.enable_monitoring = True
trainer.config.log_metrics_interval = 10  # Log every 10 episodes
```

### Performance Tracking

```python
# Get training status
status = trainer.get_training_status()
print(f"Progress: {status['progress']:.1%}")

# Get model performance
performance = trainer.get_model_performance_summary()
for model_id, stats in performance.items():
    print(f"{model_id}: {stats['avg_reward']:.3f} avg reward")
```

## Advanced Features

### VERL Integration

```python
config.enable_verl = True
config.enable_areal = True
config.kv_cache_size = 20000  # Larger cache for better performance
```

### LoRA Management

```python
from core_srl import create_distributed_lora_scheduler

lora_scheduler = create_distributed_lora_scheduler(
    base_port=8001,
    num_gpus=4
)

await lora_scheduler.start()
```

### Custom Reward Functions

```python
def custom_reward_fn(model_output, context):
    # Custom reward logic
    base_reward = len(model_output.split()) / 50.0
    quality_bonus = 0.5 if "good" in model_output.lower() else 0.0
    return base_reward + quality_bonus

# Apply to trainer
trainer.environment.reward_function = custom_reward_fn
```

## Performance Optimization

### Memory Optimization

```python
# For large models on limited hardware
config.batch_size = 16  # Smaller batches
config.episode_length = 16  # Shorter episodes
config.kv_cache_size = 5000  # Smaller cache
```

### Training Speed

```python
# Faster training
config.update_frequency = 5  # Update more frequently
config.save_interval = 200  # Save less frequently
config.log_metrics_interval = 20  # Log less frequently
```

## Example Training Session

```python
import asyncio
from core_srl import MultiModelTrainer, MultiModelConfig, TrainingMode

async def full_training_example():
    # Configure training
    config = MultiModelConfig(
        num_models=4,
        model_types=["qwen3", "openai"],
        training_mode=TrainingMode.MIXED,
        max_episodes=200,
        learning_rate=3e-4,
        checkpoint_dir="./training_checkpoints"
    )
    
    # Create trainer
    trainer = MultiModelTrainer(config)
    
    try:
        # Start training
        print("Starting multi-model training...")
        results = await trainer.train()
        
        # Analyze results
        print(f"Training completed in {results['training_time']:.2f}s")
        print(f"Total episodes: {results['total_episodes']}")
        
        # Model performance
        performance = trainer.get_model_performance_summary()
        for model_id, stats in performance.items():
            print(f"{model_id}:")
            print(f"  Avg Reward: {stats['avg_reward']:.3f}")
            print(f"  Win Rate: {stats['win_rate']:.3f}")
            print(f"  Updates: {stats['update_count']}")
        
        return results
        
    finally:
        await trainer.shutdown()

# Run the example
results = asyncio.run(full_training_example())
```

## Next Steps

- **[Model Configuration](model_config.md)** - Configure specific models
- **[Checkpoint Management](checkpoints.md)** - Advanced checkpoint features
- **[VERL/AReaL Integration](verl_areal.md)** - Performance optimization
- **[Examples](../examples/)** - More practical examples
