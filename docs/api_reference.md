# API Reference

This document provides comprehensive API documentation for Sandbox-RLX.

## 📋 Table of Contents

1. [Core Workflow](#core-workflow)
2. [LLM Interface](#llm-interface)
3. [RL Framework](#rl-framework)
4. [SandBox](#sandbox)
5. [Monitoring](#monitoring)
6. [LLM Frozen & Adaptive](#llm-frozen--adaptive)
7. [AReaL KV Cache](#areal-kv-cache)
8. [Enhanced RL Algorithms](#enhanced-rl-algorithms)

## 🔄 Core Workflow

### SG_Workflow

Main workflow class for managing DAG-based execution.

```python
from sandbox_rl.core.sg_workflow import SG_Workflow, WorkflowMode, NodeType

class SG_Workflow:
    def __init__(self, name: str, mode: WorkflowMode, llm_manager)
    
    def add_node(self, node_type: NodeType, name: str, config: Dict[str, Any])
    def add_edge(self, from_node: str, to_node: str)
    def execute_full_workflow(self) -> Dict[str, Any]
    def execute_step(self) -> Dict[str, Any]
    def get_node(self, name: str) -> EnhancedWorkflowNode
    def get_all_nodes(self) -> Dict[str, EnhancedWorkflowNode]
    def validate_workflow(self) -> bool
```

**Parameters**:
- `name`: Workflow name
- `mode`: Workflow mode (TRADITIONAL, ADVANCED)
- `llm_manager`: LLM manager instance

**Methods**:
- `add_node()`: Add a node to the workflow
- `add_edge()`: Connect two nodes
- `execute_full_workflow()`: Execute the entire workflow
- `execute_step()`: Execute one step of the workflow
- `get_node()`: Get a specific node
- `get_all_nodes()`: Get all nodes
- `validate_workflow()`: Validate workflow structure

### WorkflowMode

Enumeration of workflow modes.

```python
class WorkflowMode(Enum):
    TRADITIONAL = "traditional"
    ADVANCED = "advanced"
```

### NodeType

Enumeration of node types.

```python
class NodeType(Enum):
    SANDBOX = "sandbox"
    LLM = "llm"
    RL = "rl"
    CONDITIONAL = "conditional"
    AGGREGATOR = "aggregator"
```

### EnhancedWorkflowNode

Enhanced workflow node with advanced features.

```python
class EnhancedWorkflowNode:
    def __init__(self, name: str, node_type: NodeType, 
                 condition: NodeCondition = None,
                 limits: NodeLimits = None,
                 **kwargs)
    
    def execute(self, input_data: Any) -> Any
    def get_state(self) -> Dict[str, Any]
    def set_state(self, state: Dict[str, Any])
    def validate_input(self, input_data: Any) -> bool
    def get_metadata(self) -> Dict[str, Any]
```

**Parameters**:
- `name`: Node name
- `node_type`: Type of node
- `condition`: Execution conditions
- `limits`: Resource limits
- `**kwargs`: Additional configuration

### NodeCondition

Conditions for node execution.

```python
class NodeCondition:
    def __init__(self, max_executions: int = None,
                 time_limit: float = None,
                 success_rate_threshold: float = None)
    
    def check(self, node_state: Dict[str, Any]) -> bool
    def update(self, execution_result: Dict[str, Any])
```

### NodeLimits

Resource limits for nodes.

```python
class NodeLimits:
    def __init__(self, max_memory_mb: int = None,
                 max_cpu_percent: float = None,
                 max_execution_time: float = None)
    
    def check_limits(self, current_usage: Dict[str, Any]) -> bool
```

## 🤖 LLM Interface

### create_shared_llm_manager

Factory function to create LLM manager.

```python
from sandbox_rl.core.llm_interface import create_shared_llm_manager

def create_shared_llm_manager(
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
    backend: str = "huggingface",
    temperature: float = 0.7,
    max_tokens: int = 512,
    **kwargs
) -> SharedLLMManager
```

**Parameters**:
- `model_name`: Name of the LLM model
- `backend`: Backend to use (huggingface, openai, etc.)
- `temperature`: Sampling temperature
- `max_tokens`: Maximum tokens to generate
- `**kwargs`: Additional model parameters

### SharedLLMManager

Shared LLM manager for multi-node workflows.

```python
class SharedLLMManager:
    def __init__(self, model_name: str, backend: str, **kwargs)
    
    def register_node(self, node_name: str, config: Dict[str, Any])
    def generate_for_node(self, node_name: str, prompt: str) -> LLMResponse
    def get_model_info(self) -> Dict[str, Any]
    def update_config(self, node_name: str, config: Dict[str, Any])
    def list_registered_nodes(self) -> List[str]
```

**Methods**:
- `register_node()`: Register a node for LLM generation
- `generate_for_node()`: Generate response for specific node
- `get_model_info()`: Get model information
- `update_config()`: Update node configuration
- `list_registered_nodes()`: List all registered nodes

### LLMResponse

Response from LLM generation.

```python
class LLMResponse:
    def __init__(self, text: str, metadata: Dict[str, Any] = None)
    
    @property
    def text(self) -> str
    @property
    def metadata(self) -> Dict[str, Any]
    def to_dict(self) -> Dict[str, Any]
```

## 🎯 RL Framework

### RLTrainer

Reinforcement learning trainer.

```python
from sandbox_rl.core.rl_algorithms import RLTrainer, RLConfig

class RLTrainer:
    def __init__(self, config: RLConfig, llm_manager: SharedLLMManager = None)
    
    def add_experience(self, state: Dict[str, Any], action: str, 
                      reward: float, done: bool, next_state: Dict[str, Any] = None)
    def update_policy(self) -> Dict[str, Any]
    def get_policy(self) -> Dict[str, Any]
    def save_checkpoint(self, path: str)
    def load_checkpoint(self, path: str)
    def reset(self)
```

**Methods**:
- `add_experience()`: Add experience to replay buffer
- `update_policy()`: Update policy using collected experiences
- `get_policy()`: Get current policy
- `save_checkpoint()`: Save training checkpoint
- `load_checkpoint()`: Load training checkpoint
- `reset()`: Reset trainer state

### RLConfig

Configuration for RL trainer.

```python
class RLConfig:
    def __init__(self, algorithm: str = "PPO",
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 buffer_size: int = 10000,
                 **kwargs)
```

**Parameters**:
- `algorithm`: RL algorithm (PPO, DQN, A2C, etc.)
- `learning_rate`: Learning rate
- `batch_size`: Training batch size
- `buffer_size`: Replay buffer size

## 📦 SandBox

### SandBox

Base class for environment subsets.

```python
from sandbox_rl.core.sandbox import SandBox

class SandBox:
    def __init__(self)
    
    def execute(self, action: Any) -> Tuple[Any, float, bool]
    def get_state(self) -> Any
    def reset(self) -> Any
    def is_done(self) -> bool
    def get_action_space(self) -> Any
    def get_state_space(self) -> Any
```

**Methods**:
- `execute()`: Execute an action and return (next_state, reward, done)
- `get_state()`: Get current state
- `reset()`: Reset environment
- `is_done()`: Check if episode is done
- `get_action_space()`: Get action space
- `get_state_space()`: Get state space

## 📊 Monitoring

### SocialNetworkMonitor

Monitor for social network metrics.

```python
from sandbox_rl.core.monitoring import SocialNetworkMonitor, MonitoringConfig

class SocialNetworkMonitor:
    def __init__(self, config: MonitoringConfig)
    
    def start_monitoring(self)
    def stop_monitoring(self)
    def update_metrics(self, metrics: SocialNetworkMetrics)
    def log_metrics(self, metrics: Dict[str, Any])
    def get_current_metrics(self) -> Dict[str, Any]
    def export_metrics(self, format: str = "json") -> str
```

**Methods**:
- `start_monitoring()`: Start monitoring
- `stop_monitoring()`: Stop monitoring
- `update_metrics()`: Update metrics
- `log_metrics()`: Log metrics to backends
- `get_current_metrics()`: Get current metrics
- `export_metrics()`: Export metrics in specified format

### MonitoringConfig

Configuration for monitoring.

```python
class MonitoringConfig:
    def __init__(self, enable_wandb: bool = False,
                 enable_tensorboard: bool = False,
                 wandb_project_name: str = "sandgraph",
                 log_interval: float = 1.0,
                 alert_thresholds: Dict[str, float] = None)
```

### SocialNetworkMetrics

Metrics for social network monitoring.

```python
class SocialNetworkMetrics:
    def __init__(self, total_users: int, active_users: int,
                 engagement_rate: float, content_quality_score: float,
                 network_density: float, viral_spread_rate: float,
                 response_time_avg: float, error_rate: float)
    
    def to_dict(self) -> Dict[str, Any]
    def validate(self) -> bool
```

## 🔒 LLM Frozen & Adaptive

### FrozenAdaptiveManager

Manager for LLM parameter freezing and adaptive updates.

```python
from sandbox_rl.core.llm_frozen_adaptive import FrozenAdaptiveManager, UpdateStrategy

class FrozenAdaptiveManager:
    def __init__(self)
    
    def register_model(self, model_name: str, model_config: Dict[str, Any])
    def set_update_strategy(self, model_name: str, strategy: UpdateStrategy)
    def freeze_parameters(self, model_name: str, layer_names: List[str])
    def unfreeze_parameters(self, model_name: str, layer_names: List[str])
    def update_parameters(self, model_name: str, updates: Dict[str, Any])
    def get_model_stats(self, model_name: str) -> Dict[str, Any]
    def list_models(self) -> List[str]
    def save_checkpoint(self, model_name: str, path: str)
    def load_checkpoint(self, model_name: str, path: str)
```

**Methods**:
- `register_model()`: Register a model for management
- `set_update_strategy()`: Set update strategy for model
- `freeze_parameters()`: Freeze specific parameters
- `unfreeze_parameters()`: Unfreeze specific parameters
- `update_parameters()`: Update model parameters
- `get_model_stats()`: Get model statistics
- `list_models()`: List all managed models
- `save_checkpoint()`: Save model checkpoint
- `load_checkpoint()`: Load model checkpoint

### UpdateStrategy

Enumeration of update strategies.

```python
class UpdateStrategy(Enum):
    FROZEN = "frozen"
    ADAPTIVE = "adaptive"
    SELECTIVE = "selective"
    INCREMENTAL = "incremental"
    GRADUAL = "gradual"
```

### create_frozen_config

Factory function to create frozen configuration.

```python
def create_frozen_config(
    strategy: UpdateStrategy = UpdateStrategy.ADAPTIVE,
    learning_rate: float = 0.001,
    freeze_layers: List[str] = None,
    importance_threshold: float = 0.1
) -> Dict[str, Any]
```

## 🚀 AReaL KV Cache

### create_areal_style_trainer

Factory function to create AReaL-style trainer.

```python
from sandbox_rl.core.areal_kv_cache import create_areal_style_trainer

def create_areal_style_trainer(
    kv_cache_size: int = 5000,
    max_memory_gb: float = 4.0,
    rollout_batch_size: int = 16,
    enable_streaming: bool = True,
    cache_policy: str = "lru",
    **kwargs
) -> AReaLStyleTrainer
```

**Parameters**:
- `kv_cache_size`: Size of KV cache
- `max_memory_gb`: Maximum memory usage in GB
- `rollout_batch_size`: Batch size for rollouts
- `enable_streaming`: Enable streaming generation
- `cache_policy`: Cache replacement policy (lru, lfu, priority)

### AReaLStyleTrainer

AReaL-style trainer with KV cache optimization.

```python
class AReaLStyleTrainer:
    def __init__(self, kv_cache_size: int, max_memory_gb: float, **kwargs)
    
    def add_trajectory(self, trajectory: List[Dict[str, Any]])
    def update_policy(self, batch_size: int = None) -> Dict[str, Any]
    def get_stats(self) -> Dict[str, Any]
    def clear_cache(self)
    def set_cache_policy(self, policy: str)
    def get_cache_stats(self) -> Dict[str, Any]
```

**Methods**:
- `add_trajectory()`: Add trajectory data
- `update_policy()`: Update policy using PPO
- `get_stats()`: Get comprehensive statistics
- `clear_cache()`: Clear KV cache
- `set_cache_policy()`: Set cache replacement policy
- `get_cache_stats()`: Get cache statistics

## 🔧 Enhanced RL Algorithms

### create_enhanced_ppo_trainer

Factory function to create enhanced PPO trainer.

```python
from sandbox_rl.core.enhanced_rl_algorithms import create_enhanced_ppo_trainer

def create_enhanced_ppo_trainer(
    llm_manager: SharedLLMManager,
    learning_rate: float = 0.001,
    enable_caching: bool = True,
    cache_size: int = 10000,
    **kwargs
) -> EnhancedPPOTrainer
```

### EnhancedPPOTrainer

Enhanced PPO trainer with advanced features.

```python
class EnhancedPPOTrainer:
    def __init__(self, llm_manager: SharedLLMManager, **kwargs)
    
    def add_experience(self, state: Dict[str, Any], action: str,
                      reward: float, done: bool, next_state: Dict[str, Any] = None)
    def update_policy(self) -> Dict[str, Any]
    def get_performance_stats(self) -> Dict[str, Any]
    def enable_caching(self, cache_size: int = 10000)
    def disable_caching(self)
    def get_cache_stats(self) -> Dict[str, Any]
```

## 📝 Usage Examples

### Basic Workflow

```python
from sandbox_rl.core.llm_interface import create_shared_llm_manager
from sandbox_rl.core.sg_workflow import SG_Workflow, WorkflowMode, NodeType

# Create LLM manager
llm_manager = create_shared_llm_manager("mistralai/Mistral-7B-Instruct-v0.2")

# Create workflow
workflow = SG_Workflow("my_workflow", WorkflowMode.TRADITIONAL, llm_manager)

# Add nodes
workflow.add_node(NodeType.SANDBOX, "env", {"sandbox": MySandbox()})
workflow.add_node(NodeType.LLM, "decision", {"role": "Decision Maker"})

# Execute
result = workflow.execute_full_workflow()
```

### With Monitoring

```python
from sandbox_rl.core.monitoring import SocialNetworkMonitor, MonitoringConfig

# Create monitor
monitor = SocialNetworkMonitor(
    MonitoringConfig(
        enable_wandb=True,
        enable_tensorboard=True,
        wandb_project_name="my-project"
    )
)

# Start monitoring
monitor.start_monitoring()

# Update metrics
metrics = SocialNetworkMetrics(
    total_users=1000,
    active_users=800,
    engagement_rate=0.75,
    content_quality_score=0.8,
    network_density=0.1,
    viral_spread_rate=0.3,
    response_time_avg=1.2,
    error_rate=0.01
)
monitor.update_metrics(metrics)

# Stop monitoring
monitor.stop_monitoring()
```

### With AReaL Optimization

```python
from sandbox_rl.core.areal_kv_cache import create_areal_style_trainer

# Create AReaL trainer
trainer = create_areal_style_trainer(
    kv_cache_size=10000,
    max_memory_gb=8.0,
    rollout_batch_size=32,
    enable_streaming=True
)

# Add trajectory
trajectory = [
    {"state": {"user_count": 100}, "action": "CREATE_POST", "reward": 1.0},
    {"state": {"user_count": 101}, "action": "LIKE_POST", "reward": 0.5}
]
trainer.add_trajectory(trajectory)

# Update policy
result = trainer.update_policy(batch_size=32)
print(f"Policy loss: {result['losses']['policy_loss']:.4f}")
```

## 🆘 Error Handling

### Common Exceptions

```python
class Sandbox-RLError(Exception):
    """Base exception for Sandbox-RLX"""
    pass

class WorkflowError(Sandbox-RLError):
    """Workflow-related errors"""
    pass

class LLMError(Sandbox-RLError):
    """LLM-related errors"""
    pass

class RLError(Sandbox-RLError):
    """RL-related errors"""
    pass

class MonitoringError(Sandbox-RLError):
    """Monitoring-related errors"""
    pass
```

### Error Handling Example

```python
try:
    result = workflow.execute_full_workflow()
except WorkflowError as e:
    print(f"Workflow error: {e}")
except LLMError as e:
    print(f"LLM error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## 📚 Additional Resources

- [Quick Start Guide](quick_start_guide.md)
- [Examples Guide](examples_guide.md)
- [Monitoring Guide](monitoring_guide.md)
- [LLM Frozen & Adaptive Guide](llm_frozen_adaptive_guide.md) 