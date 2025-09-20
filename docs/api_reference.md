# API Reference - Core SRL

Complete API documentation for Core SRL multi-model RL framework.

## Core Classes

### MultiModelTrainer

Main class for multi-model RL training.

```python
class MultiModelTrainer:
    def __init__(self, config: MultiModelConfig)
    async def train(self) -> Dict[str, Any]
    async def train_multi_model_episode(self, episode_num: int) -> Dict[str, Any]
    def load_checkpoint(self, checkpoint_id: str) -> bool
    def get_training_status(self) -> Dict[str, Any]
    def get_model_performance_summary(self) -> Dict[str, Any]
    async def shutdown(self)
```

**Methods:**

- `train()`: Main training loop for all models
- `train_multi_model_episode()`: Train single episode
- `load_checkpoint()`: Resume from saved checkpoint
- `get_training_status()`: Current training progress
- `get_model_performance_summary()`: Performance metrics per model
- `shutdown()`: Clean shutdown with final checkpoint

### MultiModelConfig

Configuration for multi-model training.

```python
@dataclass
class MultiModelConfig:
    # Model configuration
    num_models: int = 4
    model_types: List[str] = ["qwen3", "openai", "claude", "llama3"]
    model_names: Dict[str, str] = {...}
    
    # Training configuration
    training_mode: TrainingMode = TrainingMode.MIXED
    weight_update_strategy: WeightUpdateStrategy = WeightUpdateStrategy.ASYNCHRONOUS
    max_episodes: int = 1000
    learning_rate: float = 3e-4
    
    # System configuration
    num_gpus: int = 4
    base_port: int = 8001
    checkpoint_dir: str = "./checkpoints/multimodel"
    
    # Optimization
    enable_verl: bool = True
    enable_areal: bool = True
```

### SharedLLMManager

Interface for managing individual LLM models.

```python
class SharedLLMManager:
    def register_node(self, node_id: str, node_config: Dict[str, Any])
    def generate_for_node(self, node_id: str, prompt: str, **kwargs) -> LLMResponse
    def update_shared_parameters(self, gradients: Dict[str, Any], learning_rate: float) -> Dict[str, Any]
    def get_global_stats(self) -> Dict[str, Any]
```

## Factory Functions

### Model Creation

```python
# Modern LLM managers
create_qwen3_manager(model_name: str, device: str = "auto") -> SharedLLMManager
create_qwen_coder_manager(model_name: str, device: str = "auto") -> SharedLLMManager  
create_qwen_math_manager(model_name: str, device: str = "auto") -> SharedLLMManager
create_openai_manager(model_name: str, api_key: str = None) -> SharedLLMManager
create_claude_manager(model_name: str, api_key: str = None) -> SharedLLMManager
create_llama3_manager(model_name: str, device: str = "auto") -> SharedLLMManager

# Multi-model trainers
create_multimodel_trainer(num_models: int, training_mode: TrainingMode) -> MultiModelTrainer
create_cooperative_multimodel_trainer(num_models: int) -> MultiModelTrainer
create_competitive_multimodel_trainer(num_models: int) -> MultiModelTrainer

# Quick start
quick_start_multimodel_training(num_models: int, max_episodes: int) -> Dict[str, Any]
```

### Environment Creation

```python
# Training environments
create_multi_model_coop_compete_env(num_models: int, cooperation_level: float) -> MultiModelTrainingEnvironment
create_multi_model_team_battle() -> MultiModelTrainingEnvironment
create_multi_model_staged_env(num_models: int) -> MultiModelTrainingEnvironment
create_maze_training_env(complexity: str) -> MultiModelTrainingEnvironment
create_social_training_env(scenario: str, num_models: int) -> MultiModelTrainingEnvironment
```

### System Components

```python
# Scheduler and resources
create_unified_scheduler(base_port: int, num_gpus: int) -> UnifiedScheduler
create_cooperative_scheduler(base_port: int, num_gpus: int) -> UnifiedScheduler
create_competitive_scheduler(base_port: int, num_gpus: int) -> UnifiedScheduler

# LoRA management
create_lora_manager(configs: Dict[int, LoRAConfig]) -> LoRAManager
create_distributed_lora_scheduler(base_port: int, num_gpus: int) -> DistributedLoRAScheduler

# Monitoring
create_unified_monitor(config: MonitoringConfig = None) -> UnifiedMonitor
create_graph_visualizer(log_file: str) -> GraphVisualizer
```

## Configuration Classes

### LLMConfig

```python
@dataclass
class LLMConfig:
    backend: LLMBackend = LLMBackend.HUGGINGFACE
    model_name: str = "Qwen/Qwen2.5-14B-Instruct"
    device: str = "auto"
    max_length: int = 32768
    temperature: float = 0.7
    api_key: Optional[str] = None
    enable_lora: bool = False
    update_strategy: UpdateStrategy = UpdateStrategy.ADAPTIVE
```

### RLConfig

```python
@dataclass
class RLConfig:
    algorithm: RLAlgorithm = RLAlgorithm.PPO
    learning_rate: float = 3e-4
    gamma: float = 0.99
    cooperation_factor: CooperationFactor
    competence_factor: CompetenceFactor
    batch_size: int = 32
```

### LoRAConfig

```python
@dataclass
class LoRAConfig:
    lora_id: int
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.1
    target_modules: List[str] = ["q_proj", "v_proj", "k_proj", "o_proj"]
    gpu_id: int = 0
    port: int = 8001
```

## Enums

### TrainingMode

```python
class TrainingMode(Enum):
    COOPERATIVE = "cooperative"    # Models help each other
    COMPETITIVE = "competitive"    # Models compete
    MIXED = "mixed"               # Dynamic cooperation/competition
    HIERARCHICAL = "hierarchical" # Leader-follower
```

### WeightUpdateStrategy

```python
class WeightUpdateStrategy(Enum):
    SYNCHRONIZED = "synchronized"  # All models update together
    ASYNCHRONOUS = "asynchronous"  # Independent updates
    FEDERATED = "federated"       # Weighted averaging
    SELECTIVE = "selective"       # Only top performers
```

### LLMBackend

```python
class LLMBackend(Enum):
    HUGGINGFACE = "huggingface"   # Local models
    OPENAI_API = "openai_api"     # OpenAI API
    ANTHROPIC = "anthropic"       # Claude API
    VLLM = "vllm"                # vLLM server
```

## Utility Functions

### Checkpoint Management

```python
# Checkpoint utilities
list_available_checkpoints(checkpoint_dir: str = "./checkpoints/multimodel") -> List[str]
load_checkpoint_metadata(checkpoint_id: str, checkpoint_dir: str) -> Dict[str, Any]

# Model information
get_available_models() -> Dict[str, List[str]]
get_version_info() -> Dict[str, str]
```

### Benchmarking

```python
# Benchmark functions
run_benchmark(runs: int, episodes: int, coop_level: float, difficulty: float) -> Dict[str, Any]

# Policy classes for benchmarking
class SimplePG:
    def act() -> Tuple[int, float]
    def update(trajectories: List[Tuple[int, float, float]])

class OurMethodPolicy:
    def act() -> Tuple[int, float]
    def observe_opponent(opp_action: int)
    def update(trajectories: List, env: CoopCompeteEnv)
```

## Usage Examples

### Basic Training

```python
import asyncio
from core_srl import quick_start_multimodel_training

# Simplest usage
results = await quick_start_multimodel_training(num_models=4, max_episodes=100)
```

### Advanced Training

```python
from core_srl import MultiModelTrainer, MultiModelConfig, TrainingMode

# Custom configuration
config = MultiModelConfig(
    num_models=6,
    training_mode=TrainingMode.MIXED,
    max_episodes=500
)

# Create and run trainer
trainer = MultiModelTrainer(config)
results = await trainer.train()

# Get final performance
performance = trainer.get_model_performance_summary()
```

### Checkpoint Management

```python
from core_srl import list_available_checkpoints, load_checkpoint_metadata

# List and load checkpoints
checkpoints = list_available_checkpoints()
metadata = load_checkpoint_metadata(checkpoints[0])

# Resume training
trainer = MultiModelTrainer(config)
trainer.load_checkpoint(checkpoints[0])
results = await trainer.train()
```

### Environment Configuration

```python
from core_srl import create_multi_model_coop_compete_env, EnvironmentType

# Create custom environment
env = create_multi_model_coop_compete_env(
    num_models=8,
    cooperation_level=0.7
)

# Use with trainer
trainer.environment = env
```

### Monitoring Setup

```python
from core_srl import create_unified_monitor, MonitoringConfig

# Configure monitoring
monitor_config = MonitoringConfig(
    enable_file_logging=True,
    log_file_path="./logs/training.json"
)

monitor = create_unified_monitor(monitor_config)
monitor.start()

# Training automatically sends metrics
trainer.config.enable_monitoring = True
```

## Error Handling

### Common Exceptions

```python
# Configuration errors
ValueError: "Unsupported LLM backend"
ValueError: "Invalid number of models"
RuntimeError: "GPU memory insufficient"

# Training errors  
asyncio.TimeoutError: "Episode timeout"
RuntimeError: "Model loading failed"
FileNotFoundError: "Checkpoint not found"

# API errors
openai.APIError: "OpenAI API error"
anthropic.APIError: "Anthropic API error"
```

### Error Recovery

```python
try:
    results = await trainer.train()
except RuntimeError as e:
    if "memory" in str(e).lower():
        # Reduce batch size and retry
        trainer.config.batch_size //= 2
        results = await trainer.train()
    else:
        raise

except asyncio.TimeoutError:
    # Save checkpoint and continue
    await trainer._save_checkpoint(trainer.current_episode)
    results = await trainer.train()
```

## Performance Optimization

### Memory Optimization

```python
# Optimize for limited memory
config = MultiModelConfig(
    batch_size=16,           # Smaller batches
    kv_cache_size=5000,      # Smaller cache
    max_checkpoints=3,       # Fewer checkpoints
    enable_areal=False       # Disable if memory constrained
)
```

### Speed Optimization

```python
# Optimize for training speed
config = MultiModelConfig(
    update_frequency=5,      # More frequent updates
    enable_verl=True,        # Enable VERL
    enable_areal=True,       # Enable AReaL
    num_gpus=8,             # Use more GPUs
    batch_size=64           # Larger batches
)
```

This API reference provides complete documentation for using Core SRL's multi-model RL training capabilities.
