# Startup Guide - How to Start LLM RL Training

Complete guide for users who want to start LLM reinforcement learning training with Core SRL.

## 🚀 Quick Start (5 Minutes)

### Step 1: Installation

```bash
# Clone repository
git clone https://github.com/your-repo/core-srl.git
cd core-srl

# Install dependencies
pip install torch transformers accelerate
pip install -r requirements.txt

# Optional: Advanced optimization
pip install verl areal anthropic openai
```

### Step 2: Basic Training

```python
import asyncio
from core_srl import quick_start_multimodel_training

# Start training immediately
async def start_training():
    results = await quick_start_multimodel_training(
        num_models=4,        # Train 4 models
        max_episodes=100     # 100 training episodes
    )
    
    print("✅ Training completed!")
    print(f"📊 Results: {results['status']}")
    
    # Show model performance
    for model_id, perf in results['model_performance'].items():
        print(f"🤖 {model_id}: {perf['avg_reward']:.3f} avg reward")
    
    return results

# Run training
results = asyncio.run(start_training())
```

### Step 3: View Results

```python
# Check saved checkpoints
from core_srl import list_available_checkpoints

checkpoints = list_available_checkpoints()
print(f"💾 Saved checkpoints: {len(checkpoints)}")
print(f"📁 Latest: {checkpoints[0] if checkpoints else 'None'}")
```

## 🎯 Focused Training Scenarios

### Scenario 1: Cooperative Training

Train models to help each other learn:

```python
from core_srl import create_cooperative_multimodel_trainer

async def cooperative_training():
    # Models share knowledge and coordinate
    trainer = create_cooperative_multimodel_trainer(num_models=4)
    
    # Configure for cooperation
    trainer.config.cooperation_strength = 0.8  # High cooperation
    trainer.config.max_episodes = 200
    
    print("🤝 Starting cooperative training...")
    results = await trainer.train()
    
    # Cooperative models should have similar performance
    performance = trainer.get_model_performance_summary()
    rewards = [p['avg_reward'] for p in performance.values()]
    variance = sum((r - sum(rewards)/len(rewards))**2 for r in rewards) / len(rewards)
    
    print(f"🎯 Cooperation success: Low variance = {variance:.4f}")
    return results

asyncio.run(cooperative_training())
```

### Scenario 2: Competitive Training

Train models to compete against each other:

```python
from core_srl import create_competitive_multimodel_trainer

async def competitive_training():
    # Models compete for performance ranking
    trainer = create_competitive_multimodel_trainer(num_models=4)
    
    # Configure for competition
    trainer.config.competition_intensity = 0.7  # High competition
    trainer.config.max_episodes = 200
    
    print("⚔️ Starting competitive training...")
    results = await trainer.train()
    
    # Show winner
    performance = trainer.get_model_performance_summary()
    winner = max(performance.items(), key=lambda x: x[1]['avg_reward'])
    
    print(f"🏆 Winner: {winner[0]} with {winner[1]['avg_reward']:.3f} reward")
    return results

asyncio.run(competitive_training())
```

### Scenario 3: Mixed Training (Recommended)

Dynamic balance of cooperation and competition:

```python
from core_srl import MultiModelTrainer, MultiModelConfig, TrainingMode

async def mixed_training():
    config = MultiModelConfig(
        num_models=6,
        training_mode=TrainingMode.MIXED,
        cooperation_strength=0.6,
        competition_intensity=0.4,
        max_episodes=300
    )
    
    trainer = MultiModelTrainer(config)
    
    print("🔄 Starting mixed training (cooperation + competition)...")
    results = await trainer.train()
    
    # Analyze balance
    performance = trainer.get_model_performance_summary()
    coop_scores = [p['cooperation_score'] for p in performance.values()]
    avg_cooperation = sum(coop_scores) / len(coop_scores)
    
    print(f"⚖️ Training balance: {avg_cooperation:.3f} cooperation score")
    return results

asyncio.run(mixed_training())
```

## 🤖 Modern Model Setup

### Using Qwen3 Models (Recommended)

```python
from core_srl import create_qwen3_manager, MultiModelConfig

# Latest Qwen3 models
config = MultiModelConfig(
    num_models=4,
    model_types=["qwen3"] * 4,
    model_names={
        "qwen3": "Qwen/Qwen2.5-14B-Instruct"  # 14B, 32K context
    }
)

# For code tasks
config.model_names["qwen3"] = "Qwen/Qwen2.5-Coder-14B-Instruct"

# For math tasks  
config.model_names["qwen3"] = "Qwen/Qwen2.5-Math-14B-Instruct"
```

### Using API Models

```python
import os

# Set API keys
os.environ["OPENAI_API_KEY"] = "your-openai-key"
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key"

# Configure API models
config = MultiModelConfig(
    num_models=3,
    model_types=["qwen3", "openai", "claude"],
    model_names={
        "qwen3": "Qwen/Qwen2.5-14B-Instruct",
        "openai": "gpt-4o-mini",                    # Cost effective
        "claude": "claude-3-5-haiku-20241022"      # Fast and efficient
    }
)
```

### Mixed Model Types

```python
# Different model families for diversity
config = MultiModelConfig(
    num_models=4,
    model_types=["qwen3", "openai", "claude", "llama3"],
    model_names={
        "qwen3": "Qwen/Qwen2.5-14B-Instruct",
        "openai": "gpt-4o-mini", 
        "claude": "claude-3-5-haiku-20241022",
        "llama3": "meta-llama/Llama-3.1-8B-Instruct"
    }
)
```

## 💾 Checkpoint Management

### Automatic Checkpointing

```python
# Training automatically saves checkpoints
config = MultiModelConfig(
    checkpoint_dir="./my_training_checkpoints",
    save_interval=50,     # Save every 50 episodes
    max_checkpoints=10    # Keep last 10 checkpoints
)

trainer = MultiModelTrainer(config)
# Checkpoints saved automatically during training
```

### Resume Training

```python
from core_srl import list_available_checkpoints

# Resume from latest checkpoint
checkpoints = list_available_checkpoints("./my_training_checkpoints")
if checkpoints:
    trainer = MultiModelTrainer(config)
    success = trainer.load_checkpoint(checkpoints[0])
    
    if success:
        print(f"📂 Resumed from episode {trainer.current_episode}")
        results = await trainer.train()  # Continue training
    else:
        print("❌ Failed to load checkpoint, starting fresh")
        results = await trainer.train()
```

## 📊 Monitoring Training

### Real-time Monitoring

```python
from core_srl import create_unified_monitor

# Setup monitoring
monitor = create_unified_monitor()
monitor.start()

# Enable monitoring in trainer
config.enable_monitoring = True
config.log_metrics_interval = 10  # Log every 10 episodes

trainer = MultiModelTrainer(config)

# Training progress will be automatically logged
results = await trainer.train()

# Get final statistics
final_stats = monitor.get_comprehensive_stats()
print(f"📈 Final stats: {final_stats}")
```

### Custom Monitoring

```python
# Monitor specific metrics
async def training_with_custom_monitoring():
    trainer = MultiModelTrainer(config)
    
    for episode in range(100):
        # Train episode
        episode_result = await trainer.train_multi_model_episode(episode)
        
        # Custom monitoring
        if episode % 10 == 0:
            status = trainer.get_training_status()
            print(f"Episode {episode}:")
            print(f"  Progress: {status['progress']:.1%}")
            print(f"  Avg Reward: {episode_result['total_reward']/config.num_models:.3f}")
            print(f"  Cooperation: {episode_result['cooperation_ratio']:.3f}")
    
    return trainer.get_model_performance_summary()
```

## 🔧 Hardware Configuration

### GPU Setup

```python
# Single GPU (for small models)
config = MultiModelConfig(
    num_models=2,
    num_gpus=1,
    model_names={"qwen3": "Qwen/Qwen2.5-7B-Instruct"}  # Smaller model
)

# Multi-GPU (recommended)
config = MultiModelConfig(
    num_models=8,
    num_gpus=4,  # 2 models per GPU
    model_names={"qwen3": "Qwen/Qwen2.5-14B-Instruct"}
)

# High-end setup
config = MultiModelConfig(
    num_models=8,
    num_gpus=8,  # 1 model per GPU
    model_names={"qwen3": "Qwen/Qwen2.5-32B-Instruct"}  # Large models
)
```

### Memory Optimization

```python
# For limited memory
config = MultiModelConfig(
    batch_size=16,          # Smaller batches
    episode_length=16,      # Shorter episodes
    kv_cache_size=5000,     # Smaller cache
    enable_areal=False      # Disable if memory tight
)

# For high memory systems
config = MultiModelConfig(
    batch_size=64,          # Larger batches
    episode_length=64,      # Longer episodes
    kv_cache_size=25000,    # Larger cache
    enable_verl=True,       # Enable optimizations
    enable_areal=True
)
```

## 🎛️ Advanced Features

### VERL/AReaL Optimization

```python
# Enable advanced optimization
config = MultiModelConfig(
    enable_verl=True,       # VERL efficient training
    enable_areal=True,      # AReaL caching
    kv_cache_size=20000     # Large cache for optimization
)

# Check optimization status
trainer = MultiModelTrainer(config)
if trainer.verl_areal_bridge:
    print("✅ VERL/AReaL optimization enabled")
    
    # Get optimization stats
    stats = trainer.verl_areal_bridge.get_training_summary()
    print(f"🚀 Optimization stats: {stats}")
```

### Custom Environments

```python
from core_srl import create_maze_training_env, create_social_training_env

# Maze navigation training
maze_env = create_maze_training_env(complexity="medium")

# Social interaction training
social_env = create_social_training_env(scenario="negotiation", num_models=6)

# Use custom environment
trainer.environment = maze_env
```

## 🔍 Troubleshooting

### Common Issues

**Out of GPU Memory:**
```python
# Solution: Use smaller models or reduce batch size
config.model_names = {"qwen3": "Qwen/Qwen2.5-7B-Instruct"}
config.batch_size = 8
```

**Training Too Slow:**
```python
# Solution: Enable optimizations
config.enable_verl = True
config.enable_areal = True
config.num_gpus = 4  # Use more GPUs
```

**API Rate Limits:**
```python
# Solution: Add delays for API models
import asyncio

class RateLimitedTrainer(MultiModelTrainer):
    async def train_multi_model_episode(self, episode_num):
        result = await super().train_multi_model_episode(episode_num)
        await asyncio.sleep(0.1)  # 100ms delay
        return result
```

**Checkpoint Loading Fails:**
```python
# Solution: Check checkpoint integrity
from core_srl import load_checkpoint_metadata

checkpoints = list_available_checkpoints()
for checkpoint_id in checkpoints:
    metadata = load_checkpoint_metadata(checkpoint_id)
    if metadata:
        print(f"✅ Valid checkpoint: {checkpoint_id}")
        break
else:
    print("❌ No valid checkpoints found, starting fresh")
```

## 📈 Performance Expectations

### Training Speed

| Setup | Models | Episodes/Hour | Memory (GB) |
|-------|--------|---------------|-------------|
| Basic | 4x7B | 100-200 | 32 |
| Standard | 4x14B | 50-100 | 64 |
| Advanced | 8x14B | 25-50 | 128 |
| Research | 8x32B | 10-25 | 256 |

### Convergence

- **Cooperative**: Fast convergence, similar performance across models
- **Competitive**: Slower convergence, diverse performance distribution  
- **Mixed**: Balanced convergence, optimal overall performance

## 💡 Tips for Success

1. **Start Small**: Begin with 2-4 models and scale up
2. **Monitor Memory**: Watch GPU memory usage closely
3. **Use Checkpoints**: Enable automatic checkpointing
4. **Enable Optimization**: Use VERL/AReaL for better performance
5. **Monitor Progress**: Enable real-time monitoring
6. **Experiment**: Try different model combinations and training modes

## 🎓 Learning Path

1. **Beginner**: Start with `quick_start_multimodel_training()`
2. **Intermediate**: Configure custom `MultiModelConfig`
3. **Advanced**: Implement custom training strategies
4. **Expert**: Integrate VERL/AReaL optimization and distributed training

## 📞 Getting Help

- Check the [API Reference](api_reference.md) for detailed documentation
- See [Examples](../examples/) for practical use cases
- Review [Troubleshooting](model_config.md#troubleshooting) for common issues

---

**You're now ready to start multi-model LLM RL training with Core SRL!** 🎉
