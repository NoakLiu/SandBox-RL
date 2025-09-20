# Quick Start Guide - Core SRL

Complete guide to get Core SRL up and running for multi-model RL training in 5 minutes.

## System Requirements

- **Python**: 3.8 or higher
- **GPU**: CUDA-capable GPU recommended (16GB+ VRAM for 14B models)
- **Memory**: 16GB+ RAM minimum
- **Storage**: 50GB+ free space for models and checkpoints

## Installation

### Option 1: Basic Installation

```bash
# Clone the repository
git clone https://github.com/NoakLiu/SandBox-RL.git
cd core-srl

# Create virtual environment
python -m venv core_srl_env
source core_srl_env/bin/activate  # On Windows: core_srl_env\Scripts\activate

# Install basic dependencies
pip install -e .
```

### Option 2: Full Installation with All Features

```bash
# Clone repository
git clone https://github.com/NoakLiu/SandBox-RL.git
cd core-srl

# Create and activate virtual environment
python -m venv core_srl_env
source core_srl_env/bin/activate

# Install with all optional dependencies
pip install -e ".[full]"
```

### Option 3: Development Installation

```bash
# For developers and contributors
git clone https://github.com/NoakLiu/SandBox-RL.git
cd core-srl

python -m venv core_srl_env
source core_srl_env/bin/activate

# Install with development tools
pip install -e ".[dev,full]"
```

## Verify Installation

Test your installation with this simple script:

```python
# test_installation.py
import asyncio
from core_srl import quick_start_multimodel_training

async def test_basic_training():
    print("Testing Core SRL installation...")
    
    try:
        # Run minimal training test
        results = await quick_start_multimodel_training(
            num_models=2,
            max_episodes=5
        )
        print("Installation successful!")
        print(f"Test results: {results['status']}")
        return True
        
    except Exception as e:
        print(f"Installation test failed: {e}")
        return False

# Run test
if __name__ == "__main__":
    success = asyncio.run(test_basic_training())
    if success:
        print("Core SRL is ready to use!")
    else:
        print("Please check your installation.")
```

## Your First Multi-Model Training

### 1. Basic Training

```python
import asyncio
from core_srl import quick_start_multimodel_training

async def basic_training():
    # Train 4 models for 50 episodes
    results = await quick_start_multimodel_training(
        num_models=4,
        max_episodes=50
    )
    
    print("Training completed!")
    print("Model Performance:")
    for model_id, perf in results['model_performance'].items():
        print(f"  {model_id}: {perf['avg_reward']:.3f} avg reward")
    
    return results

# Run training
results = asyncio.run(basic_training())
```

### 2. Cooperative Training

Train models to help each other learn:

```python
from core_srl import create_cooperative_multimodel_trainer

async def cooperative_training():
    # Models share knowledge and coordinate
    trainer = create_cooperative_multimodel_trainer(num_models=4)
    
    # Configure for cooperation
    trainer.config.cooperation_strength = 0.8  # High cooperation
    trainer.config.max_episodes = 200
    
    print("Starting cooperative training...")
    results = await trainer.train()
    
    # Cooperative models should have similar performance
    performance = trainer.get_model_performance_summary()
    rewards = [p['avg_reward'] for p in performance.values()]
    variance = sum((r - sum(rewards)/len(rewards))**2 for r in rewards) / len(rewards)
    
    print(f"Cooperation success: Low variance = {variance:.4f}")
    return results

asyncio.run(cooperative_training())
```

### 3. Competitive Training

Train models to compete against each other:

```python
from core_srl import create_competitive_multimodel_trainer

async def competitive_training():
    # Models compete for performance ranking
    trainer = create_competitive_multimodel_trainer(num_models=4)
    
    # Configure for competition
    trainer.config.competition_intensity = 0.7  # High competition
    trainer.config.max_episodes = 200
    
    print("Starting competitive training...")
    results = await trainer.train()
    
    # Show winner
    performance = trainer.get_model_performance_summary()
    winner = max(performance.items(), key=lambda x: x[1]['avg_reward'])
    
    print(f"Winner: {winner[0]} with {winner[1]['avg_reward']:.3f} reward")
    return results

asyncio.run(competitive_training())
```

### 4. Mixed Training (Recommended)

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
    
    print("Starting mixed training (cooperation + competition)...")
    results = await trainer.train()
    
    # Analyze balance
    performance = trainer.get_model_performance_summary()
    coop_scores = [p['cooperation_score'] for p in performance.values()]
    avg_cooperation = sum(coop_scores) / len(coop_scores)
    
    print(f"Training balance: {avg_cooperation:.3f} cooperation score")
    return results

asyncio.run(mixed_training())
```

## Modern Model Setup

### Using Qwen3 Models (Recommended)

```python
from core_srl import MultiModelConfig

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

### Using Specialized Models

```python
# Configure specialized open-weight models
config = MultiModelConfig(
    num_models=4,
    model_types=["qwen3", "qwen_coder", "qwen_math", "llama3"],
    model_names={
        "qwen3": "Qwen/Qwen2.5-14B-Instruct",        # General purpose
        "qwen_coder": "Qwen/Qwen2.5-Coder-14B-Instruct",  # Code specialized
        "qwen_math": "Qwen/Qwen2.5-Math-14B-Instruct",    # Math specialized
        "llama3": "meta-llama/Llama-3.1-8B-Instruct"      # Alternative architecture
    }
)
```

## Checkpoint Management

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
        print(f"Resumed from episode {trainer.current_episode}")
        results = await trainer.train()  # Continue training
    else:
        print("Failed to load checkpoint, starting fresh")
        results = await trainer.train()
```

## Hardware Configuration

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

## Environment Variables

Set up GPU configuration:

```bash
# GPU configuration
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# Optional: HuggingFace cache directory
export HF_HOME="/path/to/huggingface/cache"

# Optional: Model parallelism
export NCCL_DEBUG=INFO
```

## Troubleshooting

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

**Model Loading Issues:**
```python
# Solution: Clear cache and retry
import torch
torch.cuda.empty_cache()
# Or use smaller model variant
config.model_names = {"qwen3": "Qwen/Qwen2.5-7B-Instruct"}
```

**Import Error:**
```bash
# Make sure you're in the right environment
source core_srl_env/bin/activate
pip list | grep core-srl
```

## Next Steps

1. **Basic Training**: Start with `examples/basic_multimodel_training.py`
2. **Advanced Training**: Read `docs/multimodel_training.md`
3. **Model Configuration**: Check `docs/model_config.md`
4. **Try Examples**: Explore the `examples/` directory
5. **API Reference**: See `docs/api_reference.md`

## Getting Help

- **Documentation**: `docs/` directory
- **Examples**: `examples/` directory  
- **Issues**: https://github.com/NoakLiu/core-srl/issues

---

**You're now ready to train multiple LLMs with Core SRL!**