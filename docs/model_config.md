# Model Configuration Guide

Configure modern LLMs for multi-model RL training.

## Supported Modern Models

### Qwen 3 Series (Recommended)

```python
from core_srl import create_qwen3_manager

# General purpose - 14B parameters, 32K context
manager = create_qwen3_manager("Qwen/Qwen2.5-14B-Instruct")

# Large scale - 32B parameters
manager = create_qwen3_manager("Qwen/Qwen2.5-32B-Instruct")

# Code specialized
from core_srl import create_qwen_coder_manager
manager = create_qwen_coder_manager("Qwen/Qwen2.5-Coder-14B-Instruct")

# Math specialized  
from core_srl import create_qwen_math_manager
manager = create_qwen_math_manager("Qwen/Qwen2.5-Math-14B-Instruct")
```

### OpenAI Models

```python
from core_srl import create_openai_manager
import os

# Set API key
os.environ["OPENAI_API_KEY"] = "your-api-key"

# Latest GPT-4o (128K context)
manager = create_openai_manager("gpt-4o")

# Cost-effective option
manager = create_openai_manager("gpt-4o-mini")

# Reasoning model
manager = create_openai_manager("o1-preview")
```

### Anthropic Claude

```python
from core_srl import create_claude_manager
import os

# Set API key
os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

# Latest Claude 3.5 Sonnet (200K context)
manager = create_claude_manager("claude-3-5-sonnet-20241022")

# Fast and efficient
manager = create_claude_manager("claude-3-5-haiku-20241022")
```

### Llama 3.1

```python
from core_srl import create_llama3_manager

# 8B model (131K context)
manager = create_llama3_manager("meta-llama/Llama-3.1-8B-Instruct")

# 70B model (requires multiple GPUs)
manager = create_llama3_manager("meta-llama/Llama-3.1-70B-Instruct")
```

## Multi-Model Configurations

### Homogeneous Setup (Same Model Type)

```python
from core_srl import MultiModelConfig

# 4x Qwen3-14B for fair comparison
config = MultiModelConfig(
    num_models=4,
    model_types=["qwen3"] * 4,
    model_names={
        "qwen3": "Qwen/Qwen2.5-14B-Instruct"
    }
)
```

### Heterogeneous Setup (Different Models)

```python
# Mix of different model families
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

### Specialized Setup (Task-Specific)

```python
# Code generation training
config = MultiModelConfig(
    num_models=3,
    model_types=["qwen_coder", "openai", "llama3"],
    model_names={
        "qwen_coder": "Qwen/Qwen2.5-Coder-14B-Instruct",
        "openai": "gpt-4o",  # General coding ability
        "llama3": "meta-llama/Llama-3.1-8B-Instruct"
    }
)

# Math reasoning training
config = MultiModelConfig(
    num_models=3,
    model_types=["qwen_math", "openai", "claude"],
    model_names={
        "qwen_math": "Qwen/Qwen2.5-Math-14B-Instruct",
        "openai": "o1-preview",  # Strong reasoning
        "claude": "claude-3-5-sonnet-20241022"
    }
)
```

## Hardware Requirements

### GPU Memory Requirements

| Model Size | GPU Memory | Recommended GPU |
|------------|------------|-----------------|
| 7-8B | 16GB | RTX 4090, A100-40GB |
| 14B | 24GB | RTX 4090 (tight), A100-40GB |
| 32B | 48GB | A100-80GB, H100 |
| 70B+ | 80GB+ | Multiple A100/H100 |

### Multi-GPU Setup

```python
# 4 models across 2 GPUs
config = MultiModelConfig(
    num_models=4,
    num_gpus=2,  # 2 models per GPU
    model_names={
        "qwen3": "Qwen/Qwen2.5-14B-Instruct"  # 14B fits on modern GPUs
    }
)

# 8 models across 4 GPUs  
config = MultiModelConfig(
    num_models=8,
    num_gpus=4,  # 2 models per GPU
    base_port=8001  # Ports 8001-8008
)
```

## Training Parameters

### Learning Rates

```python
# Conservative (stable training)
config.learning_rate = 1e-4

# Standard (balanced)
config.learning_rate = 3e-4

# Aggressive (fast learning)
config.learning_rate = 1e-3
```

### Episode Configuration

```python
# Quick training
config.max_episodes = 100
config.episode_length = 16

# Standard training
config.max_episodes = 500
config.episode_length = 32

# Extensive training
config.max_episodes = 2000
config.episode_length = 64
```

### Cooperation/Competition Balance

```python
# Cooperation-focused
config.cooperation_strength = 0.8
config.competition_intensity = 0.2

# Competition-focused
config.cooperation_strength = 0.3
config.competition_intensity = 0.7

# Balanced
config.cooperation_strength = 0.5
config.competition_intensity = 0.5
```

## Performance Optimization

### Memory Optimization

```python
# Reduce memory usage
config.batch_size = 16  # Smaller batches
config.kv_cache_size = 5000  # Smaller cache
config.episode_length = 16  # Shorter episodes

# Enable optimizations
config.enable_verl = True  # VERL optimization
config.enable_areal = True  # AReaL caching
```

### Training Speed

```python
# Faster updates
config.update_frequency = 5  # Update every 5 episodes
config.save_interval = 100  # Save every 100 episodes

# Parallel processing
config.num_gpus = 4  # Use multiple GPUs
config.max_concurrent_tasks = 8  # More parallel tasks
```

## Monitoring Configuration

### Basic Monitoring

```python
config.enable_monitoring = True
config.log_metrics_interval = 10  # Log every 10 episodes
```

### Advanced Monitoring

```python
from core_srl import create_unified_monitor, MonitoringConfig

monitor_config = MonitoringConfig(
    enable_file_logging=True,
    log_file_path="./logs/training_metrics.json",
    metrics_sampling_interval=1.0
)

monitor = create_unified_monitor(monitor_config)
```

## Example Configurations

### Research Setup (High Performance)

```python
research_config = MultiModelConfig(
    num_models=8,
    model_types=["qwen3", "openai", "claude", "llama3"] * 2,
    model_names={
        "qwen3": "Qwen/Qwen2.5-32B-Instruct",  # Large models
        "openai": "gpt-4o",
        "claude": "claude-3-5-sonnet-20241022",
        "llama3": "meta-llama/Llama-3.1-70B-Instruct"
    },
    training_mode=TrainingMode.MIXED,
    weight_update_strategy=WeightUpdateStrategy.FEDERATED,
    max_episodes=2000,
    learning_rate=2e-4,
    num_gpus=8,
    enable_verl=True,
    enable_areal=True
)
```

### Development Setup (Fast Iteration)

```python
dev_config = MultiModelConfig(
    num_models=4,
    model_types=["qwen3"] * 4,
    model_names={
        "qwen3": "Qwen/Qwen2.5-7B-Instruct"  # Smaller for development
    },
    training_mode=TrainingMode.COOPERATIVE,
    max_episodes=100,
    learning_rate=5e-4,
    batch_size=16,
    save_interval=25
)
```

### Production Setup (Balanced)

```python
prod_config = MultiModelConfig(
    num_models=6,
    model_types=["qwen3", "openai", "claude"],
    model_names={
        "qwen3": "Qwen/Qwen2.5-14B-Instruct",
        "openai": "gpt-4o-mini",  # Cost-effective
        "claude": "claude-3-5-haiku-20241022"  # Fast and efficient
    },
    training_mode=TrainingMode.MIXED,
    weight_update_strategy=WeightUpdateStrategy.ASYNCHRONOUS,
    max_episodes=1000,
    learning_rate=3e-4,
    enable_monitoring=True,
    checkpoint_dir="./production_checkpoints"
)
```

## Troubleshooting

### Common Issues

**Out of Memory:**
```python
# Reduce model size or batch size
config.model_names["qwen3"] = "Qwen/Qwen2.5-7B-Instruct"
config.batch_size = 8
```

**API Rate Limits:**
```python
# Add delays for API models
import asyncio
await asyncio.sleep(0.2)  # 200ms between calls
```

**Slow Training:**
```python
# Enable optimizations
config.enable_verl = True
config.enable_areal = True
config.num_gpus = 4  # Use more GPUs
```

### Performance Tuning

**For Maximum Performance:**
- Use largest models your hardware supports
- Enable VERL and AReaL optimizations
- Use multiple GPUs with distributed training
- Increase batch size and cache size

**For Stability:**
- Use smaller learning rates (1e-4)
- Enable synchronized weight updates
- Increase save intervals for checkpoints
- Monitor memory usage closely
