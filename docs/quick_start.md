# Quick Start Guide - Core SRL

Get up and running with multi-model RL training in 5 minutes.

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM for 14B models

## Installation

```bash
# Clone and install
git clone https://github.com/NoakLiu/core-srl.git
cd core-srl
pip install -r requirements.txt

# For advanced features
pip install verl areal anthropic openai
```

## Your First Multi-Model Training

### 1. Basic Training (CPU/GPU)

```python
import asyncio
from core_srl import quick_start_multimodel_training

async def basic_training():
    # Train 4 models for 50 episodes
    results = await quick_start_multimodel_training(
        num_models=4,
        max_episodes=50
    )
    
    print("Training Results:")
    for model_id, perf in results['model_performance'].items():
        print(f"  {model_id}: {perf['avg_reward']:.3f} avg reward")
    
    return results

# Run training
results = asyncio.run(basic_training())
```

### 2. Configure API Models

```python
import os
from core_srl import create_multimodel_trainer, TrainingMode

# Set API keys
os.environ["OPENAI_API_KEY"] = "your-key"
os.environ["ANTHROPIC_API_KEY"] = "your-key"

# Create trainer with API models
trainer = create_multimodel_trainer(
    num_models=3,
    training_mode=TrainingMode.COMPETITIVE,
    model_types=["qwen3", "openai", "claude"]
)

# Start training
results = asyncio.run(trainer.train())
```

### 3. Monitor Training Progress

```python
from core_srl import create_unified_monitor

# Setup monitoring
monitor = create_unified_monitor()
monitor.start()

# Training with monitoring
trainer = create_multimodel_trainer(num_models=4)
trainer.config.enable_monitoring = True

results = asyncio.run(trainer.train())

# Check final stats
stats = monitor.get_comprehensive_stats()
print(f"Final performance: {stats}")
```

## Checkpoint Management

```python
# Training automatically saves checkpoints every 100 episodes
trainer = create_multimodel_trainer(num_models=4)
trainer.config.save_interval = 50  # Save every 50 episodes
trainer.config.checkpoint_dir = "./my_checkpoints"

# List and load checkpoints
from core_srl import list_available_checkpoints

checkpoints = list_available_checkpoints("./my_checkpoints")
if checkpoints:
    trainer.load_checkpoint(checkpoints[0])  # Load latest
```

## Next Steps

- **[Multi-Model Training Guide](multimodel_training.md)** - Advanced training strategies
- **[Model Configuration](model_config.md)** - Configure specific models
- **[Examples](../examples/)** - See practical examples
- **[API Reference](api_reference.md)** - Complete documentation

## Troubleshooting

**GPU Memory Issues:**
```python
# Use smaller models or reduce batch size
config.model_names = {
    "qwen3": "Qwen/Qwen2.5-7B-Instruct",  # Smaller model
    "openai": "gpt-4o-mini"
}
config.batch_size = 16  # Reduce batch size
```

**API Rate Limits:**
```python
# Add delays between API calls
import time
await asyncio.sleep(0.1)  # 100ms delay
```
