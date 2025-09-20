# Core SRL Quick Start Installation Guide

Get Core SRL up and running in 5 minutes for multi-model RL training.

## System Requirements

- **Python**: 3.8 or higher
- **GPU**: CUDA-capable GPU recommended (16GB+ VRAM for 14B models)
- **Memory**: 16GB+ RAM minimum
- **Storage**: 50GB+ free space for models and checkpoints

## Quick Installation

### Option 1: Basic Installation

```bash
# Clone the repository
git clone https://github.com/NoakLiu/core-srl.git
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
git clone https://github.com/NoakLiu/core-srl.git
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
git clone https://github.com/NoakLiu/core-srl.git
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

Run the test:
```bash
python test_installation.py
```

## First Training Example

Once installed, try your first multi-model training:

```python
# first_training.py
import asyncio
from core_srl import quick_start_multimodel_training

async def first_training():
    print("Starting your first multi-model training...")
    
    results = await quick_start_multimodel_training(
        num_models=4,        # Train 4 models
        max_episodes=50      # 50 training episodes
    )
    
    print("Training completed!")
    print("Model Performance:")
    for model_id, perf in results['model_performance'].items():
        print(f"  {model_id}: {perf['avg_reward']:.3f} avg reward")

# Run first training
asyncio.run(first_training())
```

## Configuration for Your Hardware

### For Limited GPU Memory (8-16GB)

```python
from core_srl import MultiModelConfig, MultiModelTrainer

# Memory-efficient configuration
config = MultiModelConfig(
    num_models=2,
    model_names={"qwen3": "Qwen/Qwen2.5-7B-Instruct"},  # Smaller model
    batch_size=8,
    kv_cache_size=5000,
    enable_areal=False  # Disable to save memory
)

trainer = MultiModelTrainer(config)
```

### For High-End Systems (32GB+ GPU)

```python
# High-performance configuration
config = MultiModelConfig(
    num_models=8,
    model_names={"qwen3": "Qwen/Qwen2.5-14B-Instruct"},
    batch_size=64,
    kv_cache_size=25000,
    enable_verl=True,
    enable_areal=True,
    num_gpus=4
)

trainer = MultiModelTrainer(config)
```

## Optional Dependencies

Install additional features as needed:

```bash
# For VERL/AReaL optimization
pip install ".[optimization]"

# For visualization
pip install ".[visualization]"

# For distributed training
pip install ".[distributed]"
```

## Environment Variables

Set up API keys if using commercial models:

```bash
# Optional: For OpenAI models
export OPENAI_API_KEY="your-openai-api-key"

# Optional: For Anthropic models  
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# GPU configuration
export CUDA_VISIBLE_DEVICES="0,1,2,3"
```

## Next Steps

1. **Basic Training**: Start with `examples/basic_multimodel_training.py`
2. **Read Documentation**: Check `docs/quick_start.md`
3. **Try Examples**: Explore the `examples/` directory
4. **Advanced Features**: Read `docs/multimodel_training.md`

## Troubleshooting

### Common Issues

**Import Error:**
```bash
# Make sure you're in the right environment
source core_srl_env/bin/activate
pip list | grep core-srl
```

**GPU Memory Error:**
```python
# Use smaller models
config.model_names = {"qwen3": "Qwen/Qwen2.5-7B-Instruct"}
config.batch_size = 4
```

**Installation Fails:**
```bash
# Update pip and try again
pip install --upgrade pip setuptools wheel
pip install -e .
```

### Getting Help

- **Documentation**: `docs/` directory
- **Examples**: `examples/` directory  
- **Issues**: https://github.com/NoakLiu/core-srl/issues

---

**You're now ready to train multiple LLMs with Core SRL!**
