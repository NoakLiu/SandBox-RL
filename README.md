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

## 📁 File Structure

```
SandGraphX/
├── sandgraph/                    # Core package directory
│   ├── core/                     # Core functional modules
│   │   ├── workflow.py          # Basic workflow implementation
│   │   ├── sg_workflow.py       # SandGraph workflow implementation
│   │   ├── dag_manager.py       # DAG graph management
│   │   ├── llm_interface.py     # LLM interface
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
│   ├── oasis_social_demo.py    # OASIS social network simulation
│   ├── enhanced_social_network_demo.py # Enhanced demo with monitoring
│   └── monitoring_example.py   # Monitoring system example
├── docs/                        # Documentation
│   └── monitoring_guide.md     # Comprehensive monitoring guide
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

### 4. Add Monitoring (Optional)
```python
from sandgraph.core.monitoring import create_monitor, MonitoringConfig
from sandgraph.core.visualization import create_visualizer

# Setup monitoring
config = MonitoringConfig(
    enable_wandb=True,
    enable_tensorboard=True,
    wandb_project_name="my-social-network"
)
monitor = create_monitor(config)
visualizer = create_visualizer("./visualizations")

# Start monitoring
monitor.start_monitoring()

# During execution, collect and update metrics
metrics = collect_metrics_from_workflow(workflow)
monitor.update_metrics(metrics)

# Stop monitoring and create visualizations
monitor.stop_monitoring()
visualizer.export_visualization_report(metrics_history)
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

<!-- ## 🚧 Future Development

For detailed monitoring documentation, see [docs/monitoring_guide.md](docs/monitoring_guide.md). -->

## 📄 License

MIT License

## 🤝 Contact

- Email - dong.liu.dl2367@yale.edu 