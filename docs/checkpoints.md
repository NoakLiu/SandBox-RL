# Checkpoint Management Guide

Complete guide for saving, loading, and managing training checkpoints in Core SRL.

## Overview

Core SRL automatically saves training checkpoints containing:
- **Model weights** for all trained models
- **LoRA parameters** for efficient adaptation
- **Training state** (episode, metrics, configuration)
- **Optimizer state** for seamless resumption

## Automatic Checkpointing

### Default Behavior

```python
from core_srl import MultiModelTrainer, MultiModelConfig

config = MultiModelConfig(
    checkpoint_dir="./checkpoints/multimodel",
    save_interval=100,  # Save every 100 episodes
    max_checkpoints=10  # Keep last 10 checkpoints
)

trainer = MultiModelTrainer(config)
# Checkpoints saved automatically during training
```

### Checkpoint Structure

```
checkpoints/multimodel/
├── multimodel_ep_100_1234567890/
│   ├── metadata.json              # Training metadata
│   ├── model_weights/             # Model parameters
│   │   ├── model_0_qwen3.json
│   │   ├── model_1_openai.json
│   │   └── model_2_claude.json
│   └── lora_weights/              # LoRA parameters
│       ├── lora_0.pt
│       ├── lora_1.pt
│       └── lora_2.pt
├── multimodel_ep_200_1234567891/
└── final_multimodel_ep_500_1234567892/  # Final checkpoint
```

## Manual Checkpoint Operations

### Save Checkpoint

```python
# Save checkpoint manually
await trainer._save_checkpoint(
    episode=trainer.current_episode,
    final=False
)

# Save final checkpoint
await trainer._save_checkpoint(
    episode=trainer.current_episode,
    final=True
)
```

### List Available Checkpoints

```python
from core_srl import list_available_checkpoints

# List all checkpoints
checkpoints = list_available_checkpoints("./checkpoints/multimodel")
print("Available checkpoints:", checkpoints)

# Output:
# ['final_multimodel_ep_500_1234567892',
#  'multimodel_ep_400_1234567891', 
#  'multimodel_ep_300_1234567890']
```

### Load Checkpoint Metadata

```python
from core_srl import load_checkpoint_metadata

# Get checkpoint information
metadata = load_checkpoint_metadata(checkpoints[0])

print(f"Episode: {metadata['episode']}")
print(f"Timestamp: {metadata['timestamp']}")
print(f"Models: {list(metadata['model_states'].keys())}")
print(f"Global metrics: {metadata['global_metrics']}")
```

### Resume Training

```python
# Resume from latest checkpoint
trainer = MultiModelTrainer(config)

checkpoints = list_available_checkpoints(config.checkpoint_dir)
if checkpoints:
    success = trainer.load_checkpoint(checkpoints[0])
    if success:
        print(f"Resumed from episode {trainer.current_episode}")
        
        # Continue training
        results = await trainer.train()
```

## Checkpoint Configuration

### Save Frequency

```python
config = MultiModelConfig(
    save_interval=50,     # Save every 50 episodes (frequent)
    save_interval=200,    # Save every 200 episodes (infrequent)
    save_interval=1,      # Save every episode (debugging)
)
```

### Retention Policy

```python
config = MultiModelConfig(
    max_checkpoints=5,    # Keep only 5 latest
    max_checkpoints=20,   # Keep 20 checkpoints
    max_checkpoints=-1,   # Keep all checkpoints (no cleanup)
)
```

### Custom Checkpoint Directory

```python
config = MultiModelConfig(
    checkpoint_dir="./my_training/checkpoints",
    # Creates: ./my_training/checkpoints/multimodel_ep_X_timestamp/
)
```

## Advanced Checkpoint Features

### Conditional Checkpointing

```python
class ConditionalTrainer(MultiModelTrainer):
    async def _save_checkpoint(self, episode: int, final: bool = False):
        # Only save if performance improved
        current_performance = sum(
            state.total_reward / max(1, state.episode_count)
            for state in self.model_states.values()
        ) / len(self.model_states)
        
        if (not hasattr(self, 'best_performance') or 
            current_performance > self.best_performance):
            self.best_performance = current_performance
            await super()._save_checkpoint(episode, final)
            print(f"New best performance: {current_performance:.3f}")
```

### Custom Metadata

```python
# Add custom information to checkpoints
class CustomTrainer(MultiModelTrainer):
    async def _save_checkpoint(self, episode: int, final: bool = False):
        # Add custom metrics before saving
        custom_metrics = {
            "convergence_rate": self._calculate_convergence(),
            "diversity_score": self._calculate_diversity(),
            "training_efficiency": self._calculate_efficiency()
        }
        
        # Save with additional metadata
        checkpoint = TrainingCheckpoint(
            checkpoint_id=f"custom_ep_{episode}_{int(time.time())}",
            timestamp=datetime.now().isoformat(),
            episode=episode,
            model_states=self.model_states.copy(),
            global_metrics={
                **self._get_standard_metrics(),
                **custom_metrics
            },
            config=self.config
        )
        
        checkpoint.save_to_disk(self.config.checkpoint_dir)
```

## Checkpoint Analysis

### Performance Comparison

```python
from core_srl import load_checkpoint_metadata
import matplotlib.pyplot as plt

# Load multiple checkpoints
checkpoints = list_available_checkpoints()
episodes = []
rewards = []

for checkpoint_id in checkpoints:
    metadata = load_checkpoint_metadata(checkpoint_id)
    if metadata:
        episodes.append(metadata['episode'])
        
        # Calculate average reward across models
        model_states = metadata['model_states']
        avg_reward = sum(
            state['total_reward'] / max(1, state['episode_count'])
            for state in model_states.values()
        ) / len(model_states)
        rewards.append(avg_reward)

# Plot training progress
plt.plot(episodes, rewards)
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.title('Training Progress Across Checkpoints')
plt.savefig('training_progress.png')
```

### Model Comparison

```python
# Compare model performance across checkpoints
def analyze_model_performance(checkpoint_id):
    metadata = load_checkpoint_metadata(checkpoint_id)
    model_stats = {}
    
    for model_id, state in metadata['model_states'].items():
        model_stats[model_id] = {
            'avg_reward': state['total_reward'] / max(1, state['episode_count']),
            'win_rate': state['win_count'] / max(1, state['episode_count']),
            'update_count': state['update_count'],
            'cooperation_score': state['cooperation_score']
        }
    
    return model_stats

# Analyze latest checkpoint
latest_stats = analyze_model_performance(checkpoints[0])
for model_id, stats in latest_stats.items():
    print(f"{model_id}: reward={stats['avg_reward']:.3f}, wins={stats['win_rate']:.3f}")
```

## Checkpoint Recovery

### Automatic Recovery

```python
def create_robust_trainer(config):
    """Create trainer with automatic checkpoint recovery"""
    trainer = MultiModelTrainer(config)
    
    # Try to resume from latest checkpoint
    checkpoints = list_available_checkpoints(config.checkpoint_dir)
    if checkpoints:
        try:
            success = trainer.load_checkpoint(checkpoints[0])
            if success:
                print(f"Resumed training from episode {trainer.current_episode}")
            else:
                print("Starting fresh training")
        except Exception as e:
            print(f"Checkpoint recovery failed: {e}")
            print("Starting fresh training")
    
    return trainer
```

### Selective Recovery

```python
# Load specific checkpoint by episode
def load_checkpoint_by_episode(target_episode, checkpoint_dir):
    checkpoints = list_available_checkpoints(checkpoint_dir)
    
    for checkpoint_id in checkpoints:
        metadata = load_checkpoint_metadata(checkpoint_id, checkpoint_dir)
        if metadata and metadata['episode'] == target_episode:
            return checkpoint_id
    
    return None

# Usage
checkpoint_id = load_checkpoint_by_episode(150, "./checkpoints/multimodel")
if checkpoint_id:
    trainer.load_checkpoint(checkpoint_id)
```

## Best Practices

### Checkpoint Naming

```python
# Use descriptive checkpoint IDs
checkpoint_id = f"qwen3_vs_gpt4o_ep_{episode}_{timestamp}"
checkpoint_id = f"cooperative_4models_ep_{episode}_{timestamp}"
checkpoint_id = f"competitive_8models_ep_{episode}_{timestamp}"
```

### Storage Management

```python
# Monitor checkpoint disk usage
import shutil

def get_checkpoint_size(checkpoint_dir):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(checkpoint_dir):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    return total_size / (1024**3)  # GB

size_gb = get_checkpoint_size("./checkpoints/multimodel")
print(f"Checkpoints using {size_gb:.2f} GB")
```

### Backup Strategy

```python
# Backup important checkpoints
import shutil

def backup_checkpoint(checkpoint_id, backup_dir):
    src = f"./checkpoints/multimodel/{checkpoint_id}"
    dst = f"{backup_dir}/{checkpoint_id}"
    shutil.copytree(src, dst)
    print(f"Backed up {checkpoint_id}")

# Backup final checkpoints
checkpoints = list_available_checkpoints()
final_checkpoints = [c for c in checkpoints if c.startswith("final_")]
for checkpoint in final_checkpoints:
    backup_checkpoint(checkpoint, "./backup/checkpoints")
```

## Integration with Training

### Checkpoint-Aware Training Loop

```python
async def robust_training_loop():
    # Create trainer with checkpoint recovery
    trainer = create_robust_trainer(config)
    
    try:
        # Training with automatic checkpointing
        results = await trainer.train()
        
        # Verify final checkpoint
        final_checkpoints = [c for c in list_available_checkpoints() 
                           if c.startswith("final_")]
        if final_checkpoints:
            print(f"Training completed. Final checkpoint: {final_checkpoints[0]}")
        
        return results
        
    except KeyboardInterrupt:
        print("Training interrupted - checkpoint saved")
        return trainer.get_training_status()
    except Exception as e:
        print(f"Training failed: {e}")
        # Emergency checkpoint
        await trainer._save_checkpoint(trainer.current_episode, final=True)
        raise
```

This checkpoint system ensures your multi-model training progress is never lost and can be resumed seamlessly.
