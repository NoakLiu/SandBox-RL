# Core SRL - Multi-Model Reinforcement Learning

<div align="center">

![Core SRL Logo](assets/logo.png)

**Advanced Multi-Model RL Framework for Training Modern LLMs**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

## 🎯 What is Core SRL?

Core SRL enables **simultaneous training of multiple modern LLMs** using reinforcement learning with **cooperative-competitive dynamics**. Train 4-8 models like Qwen3-14B, together with real-time weight updates.

### Key Features

- **Multi-Model Training**: Simultaneous RL training of 4-8 modern LLMs
- **Live Weight Updates**: Real-time parameter synchronization during training  
- **Cooperative-Competitive RL**: Novel algorithm balancing cooperation and competition
- **Modern Model Support**: Qwen3-14B, GPT-4o, Claude-3.5, Llama-3.1
- **VERL/AReaL Integration**: Efficient training with advanced caching
- **Checkpoint Management**: Automatic saving and recovery

## 🏗️ System Architecture

![System Architecture](assets/archi.jpeg)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Core SRL Architecture                        │
├─────────────────────────────────────────────────────────────────┤
│  Multi-Model Trainer                                           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐│
│  │   Qwen3-14B │ │   GPT-4o    │ │  Claude-3.5 │ │ Llama-3.1   ││
│  │   + LoRA    │ │   + LoRA    │ │   + LoRA    │ │   + LoRA    ││
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘│
│         │               │               │               │        │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │           Cooperative-Competitive RL Engine               │ │
│  │  • Weight Update Coordination  • Parameter Sharing        │ │
│  │  • VERL Integration           • AReaL Optimization        │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/your-repo/core-srl.git
cd core-srl
pip install -r requirements.txt
```

### Basic Training

```python
import asyncio
from core_srl import quick_start_multimodel_training

async def main():
    results = await quick_start_multimodel_training(
        num_models=4,
        max_episodes=100
    )
    print(f"Training completed: {results['status']}")

asyncio.run(main())
```

### Advanced Configuration

```python
from core_srl import MultiModelTrainer, MultiModelConfig, TrainingMode

config = MultiModelConfig(
    num_models=6,
    model_types=["qwen3", "openai", "claude"],
    training_mode=TrainingMode.MIXED,
    max_episodes=1000,
    checkpoint_dir="./my_checkpoints"
)

trainer = MultiModelTrainer(config)
results = asyncio.run(trainer.train())
```

### Checkpoint Management

```python
from core_srl import list_available_checkpoints

# List checkpoints
checkpoints = list_available_checkpoints()
print("Available:", checkpoints)

# Resume training
trainer.load_checkpoint(checkpoints[0])
```

## 📊 Supported Models

```python
MODERN_MODELS = {
    "qwen3": "Qwen/Qwen2.5-14B-Instruct",     # Latest Qwen
    "openai": "gpt-4o",                        # Latest GPT
    "claude": "claude-3-5-sonnet-20241022",   # Latest Claude  
    "llama3": "meta-llama/Llama-3.1-8B-Instruct"  # Latest Llama
}
```

## 📁 Project Structure

```
core-srl/
├── core_srl/           # Core framework (8 files)
├── examples/           # Training examples (8 examples)
├── tests/              # Test suites
├── docs/               # Documentation (6 docs)
├── data/               # Training data and results
└── checkpoints/        # Model checkpoints
```

## 📚 Documentation

- **[Quick Start](docs/quick_start.md)** - 5-minute setup
- **[Multi-Model Training](docs/multimodel_training.md)** - Training guide
- **[Model Configuration](docs/model_config.md)** - Modern LLM setup
- **[Checkpoints](docs/checkpoints.md)** - Save/restore training
- **[VERL/AReaL](docs/verl_areal.md)** - Advanced optimization
- **[API Reference](docs/api_reference.md)** - Complete API

## 🤝 Contributing

Focus areas:
- New modern LLM integrations
- Advanced multi-model strategies
- Performance optimizations

## 📄 License

MIT License

---

<div align="center">
<b>Core SRL v2.0.0 - Multi-Model RL Training Made Simple</b>
</div>