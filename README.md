# SandGraphX

<div align="center">
  <img src="assets/logo.png" alt="SandGraphX Logo" width="200"/>
</div>

SandGraphX is an intelligent optimization framework based on Environment Subsets abstraction and Optimization Goals. It coordinates LLM decision-making and RL weight updates through SandBox Workflow Graph to achieve automated optimization of complex tasks.

## 🌟 Core Concepts

### 1. Environment Subsets
- Decompose complex environments into manageable subsets
- Each subset is an independent SandBox
- Support custom state spaces and action spaces
- Provide standardized execution and evaluation interfaces

### 2. Optimization Goals
- Define specific optimization objectives for tasks
- Support single-objective or multi-objective optimization
- Support custom scoring functions
- Provide quantitative assessment of goal achievement

### 3. SandBox Workflow Graph
- Decompose tasks into multiple SandBox nodes
- Organize node relationships through Directed Acyclic Graph (DAG)
- Support parallel and sequential execution
- Implement state transfer and result aggregation between nodes

### 4. Intelligent Decision System
- **RL Weight Updates**: Optimize decision strategies
- **State Management**: Track and update system states
- **Isolated Interaction with LLM and Resources**: SandBox as workflow graph nodes are isolated from LLM (Decision Making), RL (LLM Weight Update), and Computational Resources (GPU, CPU, etc.), with SandGraphX globally managing the latter two.

<div align="center">
  <img src="assets/archi.jpeg" alt="SandGraphX Architecture" width="800"/>
</div>

## 🌟 Core Features

- **Sandbox Environment**: Standardized task environments following InternBootCamp patterns
- **Workflow Graph**: Support for Sandbox DAG Workflow
- **Standardized Communication**: Use official MCP protocol for Sandbox communication with LLM for computation
- **Multiple Usage Scenarios**: From single sandbox (single node) execution to complex multi-stage (multiple node, large DAGs) workflows
- **Dynamic Workflow Engine**: Support for complex DAG (Directed Acyclic Graph) workflows, enabling multi-node collaboration
- **Intelligent State Management**: Each node maintains independent states, supporting dynamic updates and state tracking
- **Resource Management System**: Resource (energy, tokens, time, knowledge) management mechanisms
- **Adaptive Decision Making**: Support for intelligent decisions based on historical information and current states
- **Extensible Architecture**: Easy to add new node types and functional modules
- **🔥 Rich LLM Model Support**: Support for various popular large language models, including:
  - **Default Recommendation**: Mistral-7B
  - **Chinese Models**: Qwen-7B, Yi-6B, ChatGLM3
  - **Code Models**: CodeLLaMA, StarCoder
  - **Lightweight**: Phi-2, Gemma-2B
  - **High Performance**: LLaMA2-13B
  - **Open Source Alternatives**: GPT-2, Falcon
- **📊 Advanced Monitoring & Visualization**: Comprehensive real-time monitoring with WanDB and TensorBoard integration
  - **Real-time Metrics Tracking**: Monitor social network metrics in real-time
  - **Multi-backend Support**: WanDB, TensorBoard, file logging, and console output
  - **Alert System**: Configurable alerts for critical thresholds
  - **Advanced Visualization**: Static dashboards, interactive plots, trend analysis, and correlation heatmaps
  - **Comprehensive Metrics**: User, engagement, content, network, community, influence, and performance metrics
- **🔒 LLMs Frozen & Adaptive Update**: Advanced parameter management for large language models
  - **Parameter Freezing**: Freeze specific layers or parameters to maintain stability
  - **Multiple Update Strategies**: FROZEN, ADAPTIVE, SELECTIVE, INCREMENTAL, GRADUAL strategies
  - **Adaptive Learning Rate**: Automatically adjust learning rates based on performance
  - **Parameter Importance Analysis**: Analyze and rank parameters by importance
  - **Performance Monitoring**: Real-time performance tracking and trend analysis
  - **Checkpoint & Rollback**: Save checkpoints and rollback to optimal states
  - **Thread-Safe Operations**: Multi-threaded parameter management with locks
- **🚀 AReaL KV Cache Optimization**: Advanced RL training optimizations based on [AReaL](https://github.com/inclusionAI/AReaL)
  - **Asynchronous RL Training**: Decoupled generation and training for improved efficiency
  - **Streaming Generation**: Real-time generation with reward computation
  - **Interruptible Rollout**: KV cache management with task interruption support
  - **Data Staleness Control**: Rollout controller with configurable staleness thresholds
  - **Decoupled PPO Loss**: Stable training with separated policy and value losses
  - **Memory-Efficient KV Cache**: Adaptive cache policies (LRU, LFU, Priority-based)
  - **Multi-threaded Processing**: Parallel rollout execution with worker pools
- **🎯 Comprehensive Misinformation Analysis**: Advanced social network analysis with multi-agent competition
  - **Multi-Agent Competition**: SandGraph LLM vs Rule-Based vs Human Simulation
  - **Real-time Performance Tracking**: Monitor agent performance and network dynamics
  - **Integrated Optimization**: Combine LLM frozen & adaptive update with AReaL KV cache
  - **WanDB Integration**: Comprehensive monitoring and visualization of competition results
  - **Network Dynamics Simulation**: Realistic social network behavior modeling
  - **Belief Impact Analysis**: Track misinformation impact on user beliefs

## 📁 File Structure

```
SandGraphX/
├── sandgraph/                    # Core package directory
│   ├── core/                     # Core functional modules
│   │   ├── workflow.py          # Basic workflow implementation
│   │   ├── sg_workflow.py       # SandGraph workflow implementation
│   │   ├── dag_manager.py       # DAG graph management
│   │   ├── llm_interface.py     # LLM interface
│   │   ├── llm_frozen_adaptive.py # LLMs frozen & adaptive update
│   │   ├── enhanced_rl_algorithms.py # Enhanced RL algorithms (Areal integration)
│   │   ├── areal_kv_cache.py      # AReaL-style KV cache optimization
│   │   ├── sandbox.py           # Sandbox base class
│   │   ├── rl_framework.py      # Reinforcement learning framework
│   │   ├── rl_algorithms.py     # Reinforcement learning algorithms
│   │   ├── monitoring.py        # Social network monitoring system
│   │   └── visualization.py     # Data visualization module
│   ├── sandbox_implementations.py # Sandbox implementations
│   └── examples.py              # Example code
├── demo/                        # Example code directory
│   ├── trading_demo.py         # Trading system example
│   ├── social_network_demo.py  # Social network analysis demo
│   ├── misinformation_spread_demo.py # Misinformation spread demo
│   ├── comprehensive_misinformation_demo.py # Comprehensive misinformation analysis with multi-agent competition
│   ├── oasis_social_demo.py    # OASIS social network simulation
│   ├── enhanced_social_network_demo.py # Enhanced demo with monitoring
│   ├── enhanced_oasis_social_demo.py   # Enhanced OASIS demo with monitoring
│   ├── enhanced_rl_cache_demo.py       # Enhanced RL cache demo
│   ├── areal_kv_cache_demo.py          # AReaL KV cache optimization demo
│   ├── monitoring_example.py   # Monitoring system example
│   ├── llm_frozen_adaptive_demo.py # LLMs frozen & adaptive demo (full)
│   └── llm_frozen_adaptive_simple_demo.py # LLMs frozen & adaptive demo (simple)
├── docs/                        # Documentation
│   ├── monitoring_guide.md     # Comprehensive monitoring guide
│   └── llm_frozen_adaptive_guide.md # LLMs frozen & adaptive guide
├── logs/                        # Log files and monitoring data
├── visualizations/              # Generated visualizations
└── setup.py                     # Installation configuration
```

## 🏗️ System Architecture

```
┌───────────────────────────────────────────────────────┐
│                      SandGraph Core                   │
├─────────────┬─────────────┬─────────────┬─────────────┤
│  Workflow   │   SandBox   │    LLM      │     RL      │
│   Engine    │  Manager    │  Manager    │  Manager    │
└──────┬──────┴──────┬──────┴──────┬──────┴──────┬──────┘
       │             │             │             │
       ▼             ▼             ▼             ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│  DAG Nodes  │ │ Environment │ │  Decision   │ │  Weight     │
│             │ │  Subsets    │ │  Making     │ │  Updates    │
└─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘
       │             │             │             │
       └─────────────┴─────────────┴─────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│                    SandGraphX Manager                   │
├─────────────────────────────────────────────────────────┤
│  • User Input: Environment subset definitions and optimization goals │
│  • Workflow: DAG graph construction and execution management         │
│  • Optimization: LLM decision optimization and RL weight updates     │
│  • Resources: Global resource management and SandBox isolation       │
│  • Monitoring: Execution state tracking and performance analysis     │
│  • Extension: Support for custom nodes and optimization strategies   │
│  • LLM Management: Frozen & adaptive parameter management            │
└─────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│                Monitoring & Visualization               │
├─────────────────────────────────────────────────────────┤
│  • Real-time Metrics: WanDB, TensorBoard, file logging │
│  • Alert System: Configurable thresholds and callbacks  │
│  • Visualization: Dashboards, trends, correlation maps │
│  • Export: JSON, CSV, images, interactive HTML         │
│  • LLM Monitoring: Parameter importance, update history │
└─────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Define Environment Subsets
```python
from sandgraph import SandBox

class MyEnvironment(SandBox):
    def __init__(self):
        super().__init__()
        self.state_space = {...}  # Define state space
        self.action_space = {...}  # Define action space
    
    def execute(self, action):
        # Implement environment execution logic
        return next_state, reward, done
    
    def get_state(self):
        # Return current state
        return self.current_state
```

### 2. Define Optimization Goals
```python
def optimization_goal(state, action, next_state):
    # Implement optimization objective function
    score = calculate_score(state, action, next_state)
    return score
```

### 3. Create Workflow
```python
from sandgraph.core.llm_interface import create_shared_llm_manager
from sandgraph.core.sg_workflow import SG_Workflow, WorkflowMode
from sandgraph.core.rl_algorithms import RLTrainer, RLConfig

# Create LLM manager (default uses Mistral-7B)
llm_manager = create_shared_llm_manager("mistralai/Mistral-7B-Instruct-v0.2")

# Create workflow
workflow = SG_Workflow("my_workflow", WorkflowMode.TRADITIONAL, llm_manager)

# Add nodes
workflow.add_node(NodeType.SANDBOX, "env", {"sandbox": MyEnvironment()})
workflow.add_node(NodeType.LLM, "decision", {"role": "Decision Maker"})
workflow.add_node(NodeType.RL, "optimizer", {"algorithm": "PPO"})

# Connect nodes
workflow.add_edge("env", "decision")
workflow.add_edge("decision", "optimizer")
workflow.add_edge("optimizer", "env")

# Execute workflow
result = workflow.execute_full_workflow()
```

### 4. AReaL KV Cache Optimization
```python
from sandgraph.core.areal_kv_cache import create_areal_style_trainer

# Create AReaL-style trainer with KV cache optimization
trainer = create_areal_style_trainer(
    kv_cache_size=10000,
    max_memory_gb=8.0,
    rollout_batch_size=32,
    enable_streaming=True
)

# Add trajectory data
trajectory = [
    {"state": {"user_count": 100}, "action": "CREATE_POST", "reward": 1.0, "ratio": 1.0, "advantage": 0.5},
    {"state": {"user_count": 101}, "action": "LIKE_POST", "reward": 0.5, "ratio": 1.1, "advantage": 0.3},
    # ... more steps
]
trainer.add_trajectory(trajectory)

# Execute decoupled PPO update
result = trainer.update_policy(batch_size=32)
print(f"Policy loss: {result['losses']['policy_loss']:.4f}")

# Get comprehensive stats
stats = trainer.get_stats()
print(f"Cache hit rate: {stats['kv_cache_stats']['hit_rate']:.3f}")
print(f"Completed tasks: {stats['rollout_stats']['completed_tasks']}")
```

### 5. Enhanced RL with Areal Integration
```python
from sandgraph.core.enhanced_rl_algorithms import create_enhanced_ppo_trainer

# Create enhanced PPO trainer with Areal framework integration
trainer = create_enhanced_ppo_trainer(
    llm_manager=llm_manager,
    learning_rate=0.001,
    enable_caching=True
)

# Add experience with caching
trainer.add_experience(
    state={"user_count": 100, "engagement": 0.3},
    action="CREATE_POST",
    reward=1.0,
    done=False
)

# Update policy with enhanced features
result = trainer.update_policy()
print(f"Cache hits: {result['cache_stats']['hits']}")
print(f"Training steps: {result['performance_stats']['training_steps']}")
```

## 📦 Installation

### Using Conda (Recommended)

```bash
# 1. Create new conda environment
conda create -n sandgraph python=3.11
conda activate sandgraph

# 2. Clone repository
git clone https://github.com/NoakLiu/SandGraphX.git
cd SandGraphX

# 3. Run installation script
chmod +x quick_install.sh
./quick_install.sh
```

### Optional Dependencies for Monitoring

```bash
# For advanced monitoring and visualization
pip install wandb tensorboard matplotlib plotly seaborn pandas

# For enhanced social network demos
pip install networkx scipy

# For LLMs frozen & adaptive update (optional)
pip install numpy
```

## 📖 Usage

### System Architecture & API Flow

```
┌─────────────────────────────────────────────────────────┐
│                    User Application                     │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│                 SandGraphX Manager                      │
│  ┌─────────────┬─────────────┬─────────────┬─────────┐  │
│  │  Workflow   │   SandBox   │    LLM      │   RL    │  │
│  │   Engine    │  Manager    │  Manager    │ Manager │  │
│  │ (sg_workflow│ (sandbox.py)│(llm_interface│(rl_algorithms│
│  │    .py)     │             │    .py)     │   .py)  │  │
│  └──────┬──────┴──────┬──────┴──────┬──────┴────┬────┘  │
│         │             │             │           │       │
│         ▼             ▼             ▼           ▼       │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────┐ ┌───────┐  │
│  │  DAG Nodes  │ │ Environment │ │ Decision│ │Weight │  │
│  │             │ │  Subsets    │ │ Making  │ │Updates│  │
│  └─────────────┘ └─────────────┘ └─────────┘ └───────┘  │
│         │             │             │           │       │
│         └─────────────┴─────────────┴───────────┘       │
│                      │                                   │
│                      ▼                                   │
│  ┌─────────────────────────────────────────────────────┐ │
│  │           LLMs Frozen & Adaptive Update             │ │
│  │        (llm_frozen_adaptive.py)                     │ │
│  │  • Parameter Freezing/Unfreezing                    │ │
│  │  • Multiple Update Strategies                       │ │
│  │  • Adaptive Learning Rate                           │ │
│  │  • Performance Monitoring                           │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│                Monitoring & Visualization               │
│  ┌─────────────┬─────────────┬─────────────┬─────────┐  │
│  │   WanDB     │ TensorBoard │   File      │ Console │  │
│  │  Logging    │   Logging   │  Logging    │ Output  │  │
│  └─────────────┴─────────────┴─────────────┴─────────┘  │
└─────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│                   Execution Results                     │
│  • Performance Metrics                                  │
│  • Optimization Statistics                              │
│  • State Updates                                        │
│  • Visualization Reports                                │
│  • LLM Parameter Analysis                               │
└─────────────────────────────────────────────────────────┘
```

### Core API Usage

```python
from sandgraph.core.llm_interface import create_shared_llm_manager
from sandgraph.core.sg_workflow import SG_Workflow, WorkflowMode
from sandgraph.core.rl_algorithms import RLTrainer, RLConfig

# 1. Initialize Core Components (default uses Mistral-7B)
llm_manager = create_shared_llm_manager(
    model_name="mistralai/Mistral-7B-Instruct-v0.2",  # Default model
    backend="huggingface",
    temperature=0.7
)

# 2. Create Workflow & RL Trainer
workflow = SG_Workflow("my_workflow", WorkflowMode.TRADITIONAL, llm_manager)
rl_trainer = RLTrainer(RLConfig(algorithm="PPO"), llm_manager)

# 3. Add Environment & Decision Nodes
workflow.add_node(NodeType.SANDBOX, "environment", {"sandbox": MySandbox()})
workflow.add_node(NodeType.LLM, "decision", {"role": "Decision Maker"})

# 4. Execute & Optimize
result = workflow.execute_full_workflow()
rl_trainer.update_policy()
```

### Example 1: Trading System

**Input**: Market data, portfolio state, trading parameters  
**Process**: LLM analyzes market → generates trading decisions → RL optimizes strategy  
**Output**: Trading actions, performance metrics, optimized weights

```python
# Run trading demo
python demo/trading_demo.py --strategy simulated --steps 5
```

### Example 2: Social Network Analysis

**Input**: Network topology, user interactions, content data  
**Process**: LLM analyzes patterns → generates insights → RL optimizes recommendations  
**Output**: Network insights, user recommendations, engagement metrics

```python
# Run social network demo
python demo/social_network_demo.py --steps 10
```

### Example 3: Enhanced Social Network with Monitoring

**Input**: Network data with comprehensive monitoring  
**Process**: LLM analysis → RL optimization → Real-time monitoring → Advanced visualization  
**Output**: Network insights, performance metrics, interactive dashboards, trend analysis

```python
# Run enhanced social network demo with monitoring
python demo/enhanced_social_network_demo.py \
    --steps 20 \
    --initial-users 100 \
    --enable-wandb \
    --enable-tensorboard \
    --wandb-project "sandgraph-enhanced-social"
```

### Example 4: Monitoring System Example

**Input**: Sample social network metrics  
**Process**: Real-time monitoring → Alert system → Multi-backend logging → Visualization  
**Output**: Comprehensive monitoring reports, interactive dashboards, trend analysis

```python
# Run monitoring example
python demo/monitoring_example.py
```

### Example 5: Misinformation Spread Analysis

**Input**: Social network data, user beliefs, information content  
**Process**: LLM analyzes misinformation patterns → generates intervention strategies → RL optimizes intervention effectiveness  
**Output**: Intervention actions, belief change metrics, spread reduction statistics

```python
# Run misinformation spread demo
python demo/misinformation_spread_demo.py --steps 5
```

### Example 6: OASIS Social Network Simulation

**Input**: User profiles, social network topology, content data  
**Process**: LLM analyzes social dynamics → generates user behaviors → RL optimizes engagement strategies  
**Output**: Social interactions, network growth metrics, engagement optimization

```python
# Run OASIS social network demo
python demo/oasis_social_demo.py --steps 5
```

### Example 7: LLMs Frozen & Adaptive Update

**Input**: LLM model, training data, performance metrics  
**Process**: Parameter importance analysis → selective freezing → adaptive updates → performance monitoring  
**Output**: Optimized model parameters, performance statistics, update history

```python
# Run simple demo (no numpy required)
python demo/llm_frozen_adaptive_simple_demo.py

# Run full demo (requires numpy)
python demo/llm_frozen_adaptive_demo.py --demo all

# Run specific demo
python demo/llm_frozen_adaptive_simple_demo.py --demo adaptive
```

### Example 8: AReaL KV Cache Optimization

**Input**: RL training data, KV cache configuration, rollout parameters  
**Process**: Asynchronous RL training → streaming generation → KV cache management → decoupled PPO updates  
**Output**: Optimized policies, cache performance metrics, training statistics

```python
# Run all AReaL optimizations
python demo/areal_kv_cache_demo.py --demo all

# Run specific components
python demo/areal_kv_cache_demo.py --demo kv_cache --cache-size 5000
python demo/areal_kv_cache_demo.py --demo rollout --batch-size 8
python demo/areal_kv_cache_demo.py --demo ppo --memory-gb 4.0
python demo/areal_kv_cache_demo.py --demo streaming --enable-streaming
```

### Example 9: Enhanced RL Cache with Areal Integration

**Input**: Enhanced RL configuration, Areal framework integration  
**Process**: Advanced caching → parallel processing → performance optimization → comprehensive monitoring  
**Output**: Enhanced training performance, cache statistics, memory efficiency metrics

```python
# Run enhanced RL cache demo
python demo/enhanced_rl_cache_demo.py \
    --demo all \
    --cache-size 10000 \
    --enable-parallel
```

### Example 10: Comprehensive Misinformation Analysis

**Input**: Social network data, multi-agent competition, integrated optimization  
**Process**: SandGraph LLM vs Rules vs Human simulation → Real-time monitoring → Integrated optimization → Performance comparison  
**Output**: Competition results, network dynamics, belief impact analysis, comprehensive metrics

```python
# Run comprehensive misinformation demo with full integration
python demo/comprehensive_misinformation_demo.py \
    --steps 50 \
    --num-users 1000 \
    --enable-wandb \
    --enable-tensorboard \
    --wandb-project "sandgraph-misinformation-competition"

# Run with custom configuration
python demo/comprehensive_misinformation_demo.py \
    --steps 100 \
    --num-users 2000 \
    --network-density 0.15 \
    --model-name "mistralai/Mistral-7B-Instruct-v0.2" \
    --kv-cache-size 10000 \
    --max-memory-gb 8.0 \
    --rollout-batch-size 32 \
    --posts-per-agent 5 \
    --enable-wandb \
    --wandb-project "sandgraph-advanced-misinformation"
```

**Key Features:**
- **Multi-Agent Competition**: SandGraph LLM competes against rule-based and human-simulated agents
- **Integrated Optimization**: Combines LLM frozen & adaptive update with AReaL KV cache optimization
- **Real-time Monitoring**: WanDB and TensorBoard integration for comprehensive tracking
- **Network Dynamics**: Realistic social network behavior with belief impact modeling
- **Performance Analysis**: Detailed comparison of agent performance in misinformation spread

**Expected Results:**
- SandGraph LLM should achieve higher misinformation spread percentage (>50%)
- Superior belief impact compared to traditional approaches
- Real-time performance tracking and visualization
- Comprehensive analysis of network dynamics and agent behavior

## 🔥 LLM Model Support

SandGraph supports various mainstream large language models. Below are the supported models and basic usage methods:

### Supported Models

| Model Type | Recommended Model | Parameter Size | Memory Requirements |
|------------|------------------|----------------|-------------------|
| **Default Recommendation** | **Mistral-7B** | 7B | 8-16GB |
| **Chinese Models** | Qwen-7B, Yi-6B, ChatGLM3 | 6-7B | 8-16GB |
| **Code Models** | CodeLLaMA, StarCoder | 7-15B | 8-16GB |
| **Lightweight** | Phi-2, Gemma-2B | 2-3B | 2-4GB |
| **High Performance** | LLaMA2-13B | 13B | 16-32GB |
| **Open Source Alternatives** | GPT-2, Falcon | 1-7B | 2-16GB |

<!-- ### LLMs Frozen & Adaptive Update Strategies

| Strategy | Use Case | Description |
|----------|----------|-------------|
| **FROZEN** | Production | Complete parameter freezing, no updates |
| **ADAPTIVE** | Development | Adaptive learning rate based on performance |
| **SELECTIVE** | Fine-tuning | Update only important parameters |
| **INCREMENTAL** | Controlled | Update parameters at fixed intervals |
| **GRADUAL** | Gradual training | Gradually reduce update intensity | -->

## 📚 Documentation

For detailed documentation on specific features:

- **Monitoring & Visualization**: [docs/monitoring_guide.md](docs/monitoring_guide.md)
- **LLMs Frozen & Adaptive Update**: [docs/llm_frozen_adaptive_guide.md](docs/llm_frozen_adaptive_guide.md)

## 📄 License

MIT License

## 🤝 Contact

- Email - dong.liu.dl2367@yale.edu 