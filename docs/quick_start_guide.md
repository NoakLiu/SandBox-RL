# Quick Start Guide

This guide will help you get up and running with SandGraphX in minutes.

## 🚀 Installation

### Prerequisites

- Python 3.11 or higher
- Conda (recommended) or pip
- Git

### Step 1: Create Environment

```bash
# Create new conda environment
conda create -n sandgraph python=3.11
conda activate sandgraph
```

### Step 2: Clone Repository

```bash
# Clone repository
git clone https://github.com/NoakLiu/SandGraphX.git
cd SandGraphX
```

### Step 3: Install Dependencies

```bash
# Run installation script
chmod +x quick_install.sh
./quick_install.sh
```

### Optional: Install Monitoring Dependencies

```bash
# For advanced monitoring and visualization
pip install wandb tensorboard matplotlib plotly seaborn pandas

# For enhanced social network demos
pip install networkx scipy

# For LLMs frozen & adaptive update (optional)
pip install numpy
```

## 🎯 Basic Usage

### 1. Define Environment Subsets

```python
from sandgraph.core.sandbox import SandBox

class MyEnvironment(SandBox):
    def __init__(self):
        super().__init__()
        self.state_space = {"user_count": 0, "engagement": 0.0}
        self.action_space = ["CREATE_POST", "LIKE_POST", "SHARE_POST"]
    
    def execute(self, action):
        # Implement environment execution logic
        if action == "CREATE_POST":
            self.state_space["user_count"] += 1
            reward = 1.0
        elif action == "LIKE_POST":
            self.state_space["engagement"] += 0.1
            reward = 0.5
        else:
            reward = 0.2
        
        return self.state_space, reward, False
    
    def get_state(self):
        return self.state_space
```

### 2. Create LLM Manager

```python
from sandgraph.core.llm_interface import create_shared_llm_manager

# Create LLM manager (default uses Mistral-7B)
llm_manager = create_shared_llm_manager(
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    backend="huggingface",
    temperature=0.7
)
```

### 3. Create Workflow

```python
from sandgraph.core.sg_workflow import SG_Workflow, WorkflowMode, NodeType

# Create workflow
workflow = SG_Workflow("my_workflow", WorkflowMode.TRADITIONAL, llm_manager)

# Add nodes
workflow.add_node(NodeType.SANDBOX, "env", {"sandbox": MyEnvironment()})
workflow.add_node(NodeType.LLM, "decision", {"role": "Decision Maker"})

# Connect nodes
workflow.add_edge("env", "decision")
workflow.add_edge("decision", "env")
```

### 4. Execute Workflow

```python
# Execute workflow
result = workflow.execute_full_workflow()
print(f"Workflow result: {result}")
```

## 🎮 Run Your First Demo

### Simple Social Network Demo

```bash
# Run a simple social network demo
python demo/social_network_demo.py --steps 10
```

This demo shows:
- Basic workflow creation
- LLM decision making
- Environment interaction
- Simple state management

### Enhanced Demo with Monitoring

```bash
# Run enhanced social network demo with monitoring
python demo/enhanced_social_network_demo.py \
    --steps 20 \
    --initial-users 100 \
    --enable-wandb \
    --enable-tensorboard \
    --wandb-project "sandgraph-enhanced-social"
```

This demo includes:
- Real-time monitoring with WanDB
- Advanced visualization
- Performance tracking
- Network dynamics simulation

## 🔧 Core Components

### Workflow Engine

The workflow engine manages the execution of your DAG (Directed Acyclic Graph) of nodes:

```python
# Add different types of nodes
workflow.add_node(NodeType.SANDBOX, "environment", {"sandbox": MySandbox()})
workflow.add_node(NodeType.LLM, "decision", {"role": "Decision Maker"})
workflow.add_node(NodeType.RL, "optimizer", {"algorithm": "PPO"})

# Connect nodes to form workflow
workflow.add_edge("environment", "decision")
workflow.add_edge("decision", "optimizer")
workflow.add_edge("optimizer", "environment")
```

### LLM Interface

The LLM interface provides a unified way to interact with different language models:

```python
# Register nodes for LLM
llm_manager.register_node("decision_maker", {"temperature": 0.8})
llm_manager.register_node("content_generator", {"temperature": 0.9})

# Generate responses
response = llm_manager.generate_for_node("decision_maker", "What should I do next?")
print(response.text)
```

### RL Framework

The RL framework provides reinforcement learning capabilities:

```python
from sandgraph.core.rl_algorithms import RLTrainer, RLConfig

# Create RL trainer
rl_trainer = RLTrainer(RLConfig(algorithm="PPO"), llm_manager)

# Add experience
rl_trainer.add_experience(
    state={"user_count": 100},
    action="CREATE_POST",
    reward=1.0,
    done=False
)

# Update policy
result = rl_trainer.update_policy()
```

## 📊 Monitoring

### Basic Monitoring

```python
from sandgraph.core.monitoring import SocialNetworkMonitor, MonitoringConfig

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
monitor.update_metrics(metrics)

# Stop monitoring
monitor.stop_monitoring()
```

### Advanced Monitoring

See the [Monitoring Guide](monitoring_guide.md) for advanced features including:
- Real-time metrics tracking
- Alert systems
- Advanced visualization
- Multi-backend support

## 🚀 Next Steps

1. **Explore Examples**: Check out the [Examples Guide](examples_guide.md) for detailed examples
2. **Learn API**: Read the [API Reference](api_reference.md) for comprehensive documentation
3. **Advanced Features**: Explore [LLM Frozen & Adaptive Guide](llm_frozen_adaptive_guide.md) for parameter management
4. **Run Demos**: Try different demos in the `demo/` directory

## 🆘 Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're in the correct conda environment
2. **Model Loading**: Check your internet connection for model downloads
3. **Memory Issues**: Reduce batch sizes or use smaller models
4. **WanDB Issues**: Make sure you're logged in with `wandb login`

### Getting Help

- Check the [API Reference](api_reference.md) for detailed documentation
- Look at the examples in the `demo/` directory
- Open an issue on GitHub for bugs or feature requests 